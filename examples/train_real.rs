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
use bibe::eval::{attribution_row, auc_roc, hit_at_k, mrr, rank_by_score_desc, Spectrum};
use bibe::model::{BibeConfig, BibeModel};
use bibe::train::{AttributionTarget, TrainConfig, Trainer};

const WINDOW: usize = 64;
const BATCH_SIZE: usize = 8;
const EPOCHS: usize = 30;

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

    let causal = std::env::args().nth(4).as_deref() == Some("causal");
    let seeds: Vec<u64> = std::env::args()
        .nth(5)
        .map(|s| s.split(',').map(|x| x.parse().expect("seed")).collect())
        .unwrap_or_else(|| vec![7, 42, 99, 1234, 2025]);

    // Spectrum FL baselines are fit on the train split's trace-level labels.
    let spectrum = Spectrum::build(train, &vocab);

    let mut runs: Vec<Vec<(String, f32)>> = Vec::new();
    for &seed in &seeds {
        println!("\n=== seed {seed} ===");
        let trainer = train_on(train, &vocab, target, object_bias, causal, seed);
        runs.push(evaluate(trainer.model(), &vocab, &spectrum, test));
    }

    println!("\n=== aggregate over {} seeds (mean ± std) ===", seeds.len());
    for (i, (name, _)) in runs[0].iter().enumerate() {
        let xs: Vec<f32> = runs.iter().map(|r| r[i].1).collect();
        let (m, s) = mean_std(&xs);
        println!("  {name:<22} {m:.3} ± {s:.3}");
    }
}

fn train_on(
    dataset: &[Trace],
    vocab: &Vocabulary,
    target: AttributionTarget,
    object_bias: f32,
    causal: bool,
    seed: u64,
) -> Trainer {
    let mut windows = Vec::new();
    for t in dataset {
        windows.extend(extract_windows(t, WINDOW, WINDOW));
    }
    let mut rng = StdRng::seed_from_u64(seed ^ 0x5eed);
    windows.shuffle(&mut rng);

    let loader = DataLoader::new(windows, BATCH_SIZE);
    let steps = loader.num_batches();

    bibe::seed(seed);
    let config = BibeConfig {
        vocab_size: vocab.len(),
        d_model: 64,
        num_heads: 4,
        d_ff: 256,
        num_layers: 2,
        n_aux: N_AUX,
        num_objects: 8,
        object_bias,
        causal,
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

fn evaluate(
    model: &BibeModel,
    vocab: &Vocabulary,
    spectrum: &Spectrum,
    test: &[Trace],
) -> Vec<(String, f32)> {
    let mut all_scores = Vec::new();
    let mut all_labels = Vec::new();
    let (mut loc1, mut loc_mrr, mut n_anom) = (0usize, 0.0f32, 0usize);
    // Attribution scored for the model and two non-learned baselines:
    // recency (most recent prior event) and same-object recency.
    let mut model_score = Scorer::default();
    let mut recency = Scorer::default();
    let mut obj_recency = Scorer::default();
    let mut obj_write_recency = Scorer::default();
    let mut trig_adjacent = Scorer::default();
    let mut trig_window = Scorer::default();
    let mut ochiai_score = Scorer::default();
    let mut tarantula_score = Scorer::default();
    let write_id = vocab.encode("write");
    let trigger_id = vocab.encode("trigger");
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

            // Same-object write recency: same-object `write` events (by recency).
            let mut obj_write_rank = candidates.clone();
            obj_write_rank.sort_by_key(|&s| {
                let same_write = sym_obj != 0
                    && batch.object_ids[s] == sym_obj
                    && batch.function_ids[s] == write_id;
                (!same_write, std::cmp::Reverse(s))
            });

            // Oracle rule for the adjacent distal generator (v1): same-object
            // writes whose immediately-preceding event is a `trigger`, by
            // recency. This is the rule the generator plants; it exposes the
            // benchmark's circularity and MUST be reported next to the model.
            let mut trig_adj_rank = candidates.clone();
            trig_adj_rank.sort_by_key(|&s| {
                let oracle = batch.function_ids[s] == write_id
                    && batch.object_ids[s] == sym_obj
                    && s > 0
                    && batch.function_ids[s - 1] == trigger_id;
                (!oracle, std::cmp::Reverse(s))
            });

            // Oracle rule for the gapped distal generator (v2): the first
            // same-object write after the first same-object trigger.
            let first_trig = (0..batch.seq).find(|&t| {
                batch.pad_mask.data[t] > 0.5
                    && batch.function_ids[t] == trigger_id
                    && batch.object_ids[t] == sym_obj
            });
            let window_oracle = first_trig.and_then(|t| {
                ((t + 1)..batch.seq).find(|&s| {
                    batch.pad_mask.data[s] > 0.5
                        && batch.function_ids[s] == write_id
                        && batch.object_ids[s] == sym_obj
                })
            });
            let mut trig_window_rank = candidates.clone();
            trig_window_rank
                .sort_by_key(|&s| (Some(s) != window_oracle, std::cmp::Reverse(s)));

            // Spectrum FL baselines: rank candidates by the suspiciousness of
            // their function (train-split coverage stats), recency tie-break.
            let mut ochiai_rank = candidates.clone();
            ochiai_rank.sort_by(|&a, &b| {
                spectrum
                    .ochiai(batch.function_ids[b])
                    .partial_cmp(&spectrum.ochiai(batch.function_ids[a]))
                    .unwrap()
                    .then(b.cmp(&a))
            });
            let mut tarantula_rank = candidates.clone();
            tarantula_rank.sort_by(|&a, &b| {
                spectrum
                    .tarantula(batch.function_ids[b])
                    .partial_cmp(&spectrum.tarantula(batch.function_ids[a]))
                    .unwrap()
                    .then(b.cmp(&a))
            });
            ochiai_score.add(&ochiai_rank, cause);
            tarantula_score.add(&tarantula_rank, cause);

            model_score.add(&model_rank, cause);
            recency.add(&rec_rank, cause);
            obj_recency.add(&obj_rank, cause);
            obj_write_recency.add(&obj_write_rank, cause);
            trig_adjacent.add(&trig_adj_rank, cause);
            trig_window.add(&trig_window_rank, cause);
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
        println!("    same-obj write     {}", obj_write_recency.report(n_att));
        println!("    trig-adjacent      {}", trig_adjacent.report(n_att));
        println!("    trig-window        {}", trig_window.report(n_att));
        println!("    ochiai FL          {}", ochiai_score.report(n_att));
        println!("    tarantula FL       {}", tarantula_score.report(n_att));
    }

    let mut metrics = vec![("detection_auc".to_string(), auc)];
    if n_anom > 0 {
        metrics.push(("localize_hit1".to_string(), loc1 as f32 / n_anom as f32));
        metrics.push(("localize_mrr".to_string(), loc_mrr / n_anom as f32));
    }
    if n_att > 0 {
        let n = n_att as f32;
        for (name, s) in [
            ("model", &model_score),
            ("recency", &recency),
            ("obj_recency", &obj_recency),
            ("obj_write", &obj_write_recency),
            ("trig_adjacent", &trig_adjacent),
            ("trig_window", &trig_window),
            ("ochiai", &ochiai_score),
            ("tarantula", &tarantula_score),
        ] {
            metrics.push((format!("{name}_hit1"), s.hit1 as f32 / n));
            metrics.push((format!("{name}_hit3"), s.hit3 as f32 / n));
            metrics.push((format!("{name}_mrr"), s.mrr / n));
        }
    }
    metrics
}

fn mean_std(xs: &[f32]) -> (f32, f32) {
    let n = xs.len() as f32;
    let m = xs.iter().sum::<f32>() / n;
    let v = xs.iter().map(|x| (x - m).powi(2)).sum::<f32>() / n;
    (m, v.sqrt())
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
