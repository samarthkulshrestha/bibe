//! Train and evaluate on real captured traces.
//!
//! Loads BiBE traces produced by `instrumentation/capture_corpus.sh` (real
//! executions labeled by AddressSanitizer), splits train/test, trains the
//! model, and reports detection, localization, and attribution against the
//! sanitizer-derived ground truth. In this corpus clean and buggy runs share
//! the same tokens — only the order of free vs use differs — so detection must
//! come from context, not a per-event feature.
//!
//! ```text
//! cargo run --release --example train_real -- <traces_dir>
//! ```

use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;

use bibe::autograd::Var;
use bibe::data::{
    collate, extract_windows, parse_trace_file, DataLoader, Trace, Vocabulary, N_AUX,
};
use bibe::eval::{attribution_row, auc_roc, hit_at_k, mrr, rank_by_score_desc};
use bibe::model::{BibeConfig, BibeModel};
use bibe::train::{AttributionTarget, TrainConfig, Trainer};

const WINDOW: usize = 64;
const BATCH_SIZE: usize = 8;
const EPOCHS: usize = 30;
const SEED: u64 = 1234;

fn main() {
    let dir = std::env::args().nth(1).expect("usage: train_real <traces_dir> [raw|rollout|margin]");
    let target = match std::env::args().nth(2).as_deref() {
        Some("rollout") => AttributionTarget::Rollout,
        Some("margin") => AttributionTarget::Margin,
        _ => AttributionTarget::RawAttention,
    };
    // Optional 3rd arg: object-aware attention bias strength (default 4, which
    // makes attribution near-perfect; pass 0 to disable).
    let object_bias: f32 = std::env::args().nth(3).and_then(|s| s.parse().ok()).unwrap_or(4.0);
    println!("attribution supervision target: {target:?}, object_bias: {object_bias}");

    // Load and deterministically order all captured traces.
    let mut paths: Vec<_> = std::fs::read_dir(&dir)
        .expect("read traces dir")
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "trace"))
        .collect();
    paths.sort();
    let traces: Vec<Trace> = paths
        .iter()
        .map(|p| parse_trace_file(p).expect("parse trace"))
        .collect();
    assert!(!traces.is_empty(), "no .trace files in {dir}");

    // 80/20 train/test split.
    let split = traces.len() * 4 / 5;
    let (train, test) = traces.split_at(split);
    let anom = train.iter().filter(|t| t.is_anomalous()).count();
    println!(
        "{} traces: {} train ({anom} anomalous), {} test",
        traces.len(),
        train.len(),
        test.len()
    );

    let vocab = Vocabulary::build(train, 1);
    println!("vocabulary size: {}", vocab.len());

    let trainer = train_on(train, &vocab, target, object_bias);
    evaluate(trainer.model(), &vocab, test);
}

fn train_on(dataset: &[Trace], vocab: &Vocabulary, target: AttributionTarget, object_bias: f32) -> Trainer {
    let mut windows = Vec::new();
    for t in dataset {
        windows.extend(extract_windows(t, WINDOW, WINDOW));
    }
    let mut rng = StdRng::seed_from_u64(7);
    windows.shuffle(&mut rng);

    let loader = DataLoader::new(windows, BATCH_SIZE);
    let steps = loader.num_batches();

    bibe::seed(SEED);
    let config = BibeConfig {
        vocab_size: vocab.len(),
        d_model: 64,
        num_heads: 4,
        d_ff: 256,
        num_layers: 2,
        n_aux: N_AUX,
        num_objects: 8,
        object_bias,
        max_len: WINDOW,
        dropout_p: 0.0,
    };
    let train_cfg = TrainConfig {
        lr: 1e-3,
        warmup_steps: steps,
        total_steps: EPOCHS * steps,
        grad_clip: 1.0,
        attribution_lambda: 1.0,
        attribution_target: target,
        ..TrainConfig::default()
    };
    let mut trainer = Trainer::new(BibeModel::new(&config), train_cfg);
    println!("\ntraining {EPOCHS} epochs x {steps} batches:");
    for epoch in 0..EPOCHS {
        let loss = trainer.train_epoch(&loader, vocab);
        if epoch % 5 == 0 || epoch == EPOCHS - 1 {
            println!("  epoch {epoch:>2}  mean loss {loss:.4}");
        }
    }
    trainer
}

fn evaluate(model: &BibeModel, vocab: &Vocabulary, test: &[Trace]) {
    let mut all_scores = Vec::new();
    let mut all_labels = Vec::new();
    let (mut loc1, mut loc_mrr, mut n_anom) = (0usize, 0.0f32, 0usize);
    // Attribution scored for the model and two non-learned baselines:
    // recency (most recent prior event) and same-object recency.
    let mut model_score = Scorer::default();
    let mut recency = Scorer::default();
    let mut obj_recency = Scorer::default();
    let mut n_att = 0usize;

    for trace in test {
        let windows = extract_windows(trace, WINDOW, WINDOW);
        let batch = collate(&windows[..1], vocab);
        let aux = Var::new(batch.aux.clone(), false);
        let out = model.forward(&batch.function_ids, &batch.object_ids, &aux, 1, batch.seq, false);
        let scores = out.anomaly_scores.tensor().data;

        // Detection + localization over real positions.
        let mut win_scores = Vec::new();
        let mut symptom_local = None;
        for (s, &score) in scores.iter().enumerate() {
            if batch.pad_mask.data[s] > 0.5 {
                let is_pos = batch.labels.data[s] > 0.5;
                if is_pos {
                    symptom_local = Some(win_scores.len());
                }
                win_scores.push(score);
                all_scores.push(score);
                all_labels.push(is_pos);
            }
        }
        if let Some(sym) = symptom_local {
            let ranked = rank_by_score_desc(&win_scores);
            loc1 += hit_at_k(&ranked, sym, 1) as usize;
            loc_mrr += mrr(&ranked, sym);
            n_anom += 1;
        }

        // Attribution: rank candidate source events and find the cause, for the
        // model and the two baselines.
        if let (Some(crash), Some(cause)) = (trace.root_cause(), trace.cause())
            && cause != crash
        {
            let candidates: Vec<usize> = (0..batch.seq)
                .filter(|&s| batch.pad_mask.data[s] > 0.5 && s != crash)
                .collect();

            // Model: rank by the attribution map.
            let row = attribution_row(&out.attribution, 0, crash);
            let mut model_rank = candidates.clone();
            model_rank.sort_by(|&a, &b| row[b].partial_cmp(&row[a]).unwrap());

            // Recency: most recent (highest position) prior event first.
            let mut rec_rank = candidates.clone();
            rec_rank.sort_by(|&a, &b| b.cmp(&a));

            // Same-object recency: same-object events (by recency) first.
            let sym_obj = batch.object_ids[crash];
            let mut obj_rank = candidates.clone();
            obj_rank.sort_by_key(|&s| {
                let same = sym_obj != 0 && batch.object_ids[s] == sym_obj;
                (!same, std::cmp::Reverse(s))
            });

            model_score.add(&model_rank, cause);
            recency.add(&rec_rank, cause);
            obj_recency.add(&obj_rank, cause);
            n_att += 1;
        }
    }

    let auc = auc_roc(&all_scores, &all_labels);
    println!("\nheld-out evaluation on real traces:");
    println!("  detection   AUC-ROC   {auc:.3}   ({} events)", all_scores.len());
    if n_anom > 0 {
        println!("  localize    Hit@1     {:.3}", loc1 as f32 / n_anom as f32);
        println!("  localize    MRR       {:.3}   ({n_anom} anomalous)", loc_mrr / n_anom as f32);
    }
    if n_att > 0 {
        println!("  attribution over {n_att} use-after-free (Hit@1 / Hit@3 / MRR):");
        println!("    model              {}", model_score.report(n_att));
        println!("    recency baseline   {}", recency.report(n_att));
        println!("    same-obj recency   {}", obj_recency.report(n_att));
    }
}

/// Accumulates Hit@1 / Hit@3 / MRR over a set of rankings.
#[derive(Default)]
struct Scorer {
    hit1: usize,
    hit3: usize,
    mrr: f32,
}

impl Scorer {
    fn add(&mut self, ranked: &[usize], target: usize) {
        self.hit1 += hit_at_k(ranked, target, 1) as usize;
        self.hit3 += hit_at_k(ranked, target, 3) as usize;
        self.mrr += mrr(ranked, target);
    }

    fn report(&self, n: usize) -> String {
        let n = n as f32;
        format!("{:.3}   {:.3}   {:.3}", self.hit1 as f32 / n, self.hit3 as f32 / n, self.mrr / n)
    }
}
