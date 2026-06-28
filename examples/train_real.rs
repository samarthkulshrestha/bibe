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
use bibe::train::{TrainConfig, Trainer};

const WINDOW: usize = 64;
const BATCH_SIZE: usize = 8;
const EPOCHS: usize = 30;
const SEED: u64 = 1234;

fn main() {
    let dir = std::env::args().nth(1).expect("usage: train_real <traces_dir>");

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

    let trainer = train_on(train, &vocab);
    evaluate(trainer.model(), &vocab, test);
}

fn train_on(dataset: &[Trace], vocab: &Vocabulary) -> Trainer {
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
        max_len: WINDOW,
        dropout_p: 0.0,
    };
    let train_cfg = TrainConfig {
        lr: 1e-3,
        warmup_steps: steps,
        total_steps: EPOCHS * steps,
        grad_clip: 1.0,
        attribution_lambda: 1.0,
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
    let (mut att1, mut att3, mut att_mrr, mut n_att) = (0usize, 0usize, 0.0f32, 0usize);

    for trace in test {
        let windows = extract_windows(trace, WINDOW, WINDOW);
        let batch = collate(&windows[..1], vocab);
        let aux = Var::new(batch.aux.clone(), false);
        let out = model.forward(&batch.function_ids, &aux, 1, batch.seq, false);
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

        // Attribution: symptom -> cause via rollout, when the cause is distinct.
        if let (Some(crash), Some(cause)) = (trace.root_cause(), trace.cause())
            && cause != crash
        {
            let row = attribution_row(&out.attribution, 0, crash);
            let mut ranked: Vec<usize> = (0..batch.seq)
                .filter(|&s| batch.pad_mask.data[s] > 0.5 && s != crash)
                .collect();
            ranked.sort_by(|&a, &b| row[b].partial_cmp(&row[a]).unwrap());
            att1 += hit_at_k(&ranked, cause, 1) as usize;
            att3 += hit_at_k(&ranked, cause, 3) as usize;
            att_mrr += mrr(&ranked, cause);
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
        println!("  attribute   Hit@1     {:.3}", att1 as f32 / n_att as f32);
        println!("  attribute   Hit@3     {:.3}", att3 as f32 / n_att as f32);
        println!("  attribute   MRR       {:.3}   ({n_att} use-after-free)", att_mrr / n_att as f32);
    }
}
