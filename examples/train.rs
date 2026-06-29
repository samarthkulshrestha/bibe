//! End-to-end training, evaluation, and Stage-A stress tests on synthetic data.
//!
//! Trains the BiBE model with the full loss (focal + sparsity + contrastive +
//! attention attribution supervision) and then runs three checks:
//!   * in-distribution detection and localization metrics,
//!   * an attribution experiment — does attention link a crash to its cause?
//!   * an out-of-distribution test — train without use-after-free, detect it.
//!
//! ```text
//! cargo run --release --example train
//! ```

use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;

use bibe::autograd::Var;
use bibe::data::{
    collate, extract_windows, BugKind, DataLoader, Trace, TraceGenerator, Vocabulary, N_AUX,
};
use bibe::eval::{attribution_row, auc_roc, hit_at_k, mrr, precision_at_k, rank_by_score_desc};
use bibe::model::{BibeConfig, BibeModel};
use bibe::train::{save_parameters, TrainConfig, Trainer};

const WINDOW: usize = 64;
const BATCH_SIZE: usize = 8;
const EPOCHS: usize = 20;
const SEED: u64 = 1234;

fn main() {
    // In-distribution: train on every bug kind with attribution supervision on.
    let dataset = TraceGenerator::new(42).dataset(60, 30);
    let n_anom = dataset.iter().filter(|t| t.is_anomalous()).count();
    println!("generated {} traces ({n_anom} anomalous)", dataset.len());

    let vocab = Vocabulary::build(&dataset, 1);
    println!("vocabulary size: {}", vocab.len());

    let trainer = train_on(&dataset, &vocab, true);

    let eval_set = TraceGenerator::new(99).dataset(20, 20);
    evaluate(trainer.model(), &vocab, &eval_set);
    attribution_experiment(trainer.model(), &vocab);
    ood_experiment();

    let path = std::path::Path::new("bibe_checkpoint.bin");
    save_parameters(path, &trainer.model().parameters()).expect("failed to save checkpoint");
    println!("\nsaved checkpoint to {}", path.display());
}

/// Window, shuffle, and train a model on the dataset; returns the trainer.
fn train_on(dataset: &[Trace], vocab: &Vocabulary, verbose: bool) -> Trainer {
    let mut windows = Vec::new();
    for t in dataset {
        windows.extend(extract_windows(t, WINDOW, WINDOW));
    }
    let mut rng = StdRng::seed_from_u64(7);
    windows.shuffle(&mut rng);

    let loader = DataLoader::new(windows, BATCH_SIZE);
    let steps_per_epoch = loader.num_batches();

    // Seed weight init so the whole run is reproducible.
    bibe::seed(SEED);
    let config = BibeConfig {
        vocab_size: vocab.len(),
        d_model: 64,
        num_heads: 4,
        d_ff: 256,
        num_layers: 2,
        n_aux: N_AUX,
        num_objects: 8,
        object_bias: 0.0,
        max_len: WINDOW,
        dropout_p: 0.0,
    };
    let train_cfg = TrainConfig {
        lr: 1e-3,
        warmup_steps: steps_per_epoch,
        total_steps: EPOCHS * steps_per_epoch,
        grad_clip: 1.0,
        attribution_lambda: 1.0,
        ..TrainConfig::default()
    };
    let mut trainer = Trainer::new(BibeModel::new(&config), train_cfg);

    if verbose {
        println!("\ntraining {EPOCHS} epochs x {steps_per_epoch} batches:");
    }
    let mut first = None;
    for epoch in 0..EPOCHS {
        let loss = trainer.train_epoch(&loader, vocab);
        if first.is_none() {
            first = Some(loss);
        }
        if verbose {
            println!("  epoch {epoch:>2}  mean loss {loss:.4}");
        }
    }
    trainer
}

/// Pooled event-level anomaly-detection AUC over a held-out set.
fn detection_auc(model: &BibeModel, vocab: &Vocabulary, eval_set: &[Trace]) -> f32 {
    let mut scores = Vec::new();
    let mut labels = Vec::new();
    for trace in eval_set {
        for window in extract_windows(trace, WINDOW, WINDOW) {
            let batch = collate(&[window], vocab);
            let aux = Var::new(batch.aux.clone(), false);
            let out = model.forward(&batch.function_ids, &batch.object_ids, &aux, 1, batch.seq, false);
            let s = out.anomaly_scores.tensor().data;
            for (i, &score) in s.iter().enumerate() {
                if batch.pad_mask.data[i] > 0.5 {
                    scores.push(score);
                    labels.push(batch.labels.data[i] > 0.5);
                }
            }
        }
    }
    auc_roc(&scores, &labels)
}

/// Detection and localization metrics against the generator's root causes.
fn evaluate(model: &BibeModel, vocab: &Vocabulary, eval_set: &[Trace]) {
    let mut all_scores = Vec::new();
    let mut all_labels = Vec::new();
    let (mut hits1, mut hits5, mut mrr_sum, mut n_anom) = (0usize, 0usize, 0.0f32, 0usize);

    for trace in eval_set {
        for window in extract_windows(trace, WINDOW, WINDOW) {
            let batch = collate(&[window], vocab);
            let aux = Var::new(batch.aux.clone(), false);
            let out = model.forward(&batch.function_ids, &batch.object_ids, &aux, 1, batch.seq, false);
            let scores = out.anomaly_scores.tensor().data;

            let mut win_scores = Vec::new();
            let mut root_cause_local = None;
            for (s, &score) in scores.iter().enumerate() {
                if batch.pad_mask.data[s] > 0.5 {
                    let is_pos = batch.labels.data[s] > 0.5;
                    if is_pos {
                        root_cause_local = Some(win_scores.len());
                    }
                    win_scores.push(score);
                    all_scores.push(score);
                    all_labels.push(is_pos);
                }
            }

            if let Some(rc) = root_cause_local {
                let ranked = rank_by_score_desc(&win_scores);
                hits1 += hit_at_k(&ranked, rc, 1) as usize;
                hits5 += hit_at_k(&ranked, rc, 5) as usize;
                mrr_sum += mrr(&ranked, rc);
                n_anom += 1;
            }
        }
    }

    let n_pos = all_labels.iter().filter(|&&l| l).count();
    let auc = auc_roc(&all_scores, &all_labels);
    let p_at = precision_at_k(&all_scores, &all_labels, n_pos.max(1));

    println!("\nheld-out evaluation ({n_anom} anomalous windows, {} events):", all_scores.len());
    println!("  detection   AUC-ROC          {auc:.3}");
    println!("  detection   Precision@{n_pos:<3}    {p_at:.3}");
    if n_anom > 0 {
        println!("  localize    Hit@1            {:.3}", hits1 as f32 / n_anom as f32);
        println!("  localize    Hit@5            {:.3}", hits5 as f32 / n_anom as f32);
        println!("  localize    MRR              {:.3}", mrr_sum / n_anom as f32);
    }
}

/// Attribution: for use-after-free traces, rank events by how much the crash
/// attributes to them via attention rollout, and check whether the causal
/// `free` surfaces.
fn attribution_experiment(model: &BibeModel, vocab: &Vocabulary) {
    let mut generator = TraceGenerator::new(777);
    let (mut hit1, mut hit3, mut mrr_sum, mut count, mut cand_total) = (0, 0, 0.0, 0, 0);

    for _ in 0..40 {
        let trace = generator.anomalous_trace(BugKind::UseAfterFree);
        let crash = trace.root_cause().unwrap();
        let free_idx = match trace.cause() {
            Some(c) if c != crash => c,
            _ => continue,
        };

        let windows = extract_windows(&trace, WINDOW, WINDOW);
        let batch = collate(&windows[..1], vocab);
        let aux = Var::new(batch.aux.clone(), false);
        let out = model.forward(&batch.function_ids, &batch.object_ids, &aux, 1, batch.seq, false);

        let row = attribution_row(&out.attribution, 0, crash);
        let mut ranked: Vec<usize> = (0..batch.seq)
            .filter(|&s| batch.pad_mask.data[s] > 0.5 && s != crash)
            .collect();
        ranked.sort_by(|&a, &b| row[b].partial_cmp(&row[a]).unwrap());

        cand_total += ranked.len();
        hit1 += hit_at_k(&ranked, free_idx, 1) as usize;
        hit3 += hit_at_k(&ranked, free_idx, 3) as usize;
        mrr_sum += mrr(&ranked, free_idx);
        count += 1;
    }

    if count == 0 {
        return;
    }
    let avg_cands = cand_total as f32 / count as f32;
    println!("\nattribution: crash -> free over {count} use-after-free traces:");
    println!("  Hit@1   {:.3}   (random ~ {:.3})", hit1 as f32 / count as f32, 1.0 / avg_cands);
    println!("  Hit@3   {:.3}   (random ~ {:.3})", hit3 as f32 / count as f32, 3.0 / avg_cands);
    println!("  MRR     {:.3}", mrr_sum / count as f32);
}

/// Out-of-distribution: train without use-after-free, then detect it.
fn ood_experiment() {
    println!("\n--- out-of-distribution: train without use-after-free, test on it ---");
    let kinds = [BugKind::Deadlock, BugKind::MemoryLeak, BugKind::PerfRegression];
    let train_set = TraceGenerator::new(13).dataset_with_kinds(60, 30, &kinds);
    let vocab = Vocabulary::build(&train_set, 1);
    let trainer = train_on(&train_set, &vocab, false);

    let eval_set = TraceGenerator::new(57).dataset_with_kinds(0, 20, &[BugKind::UseAfterFree]);
    let auc = detection_auc(trainer.model(), &vocab, &eval_set);
    println!("  detection AUC-ROC on unseen use-after-free: {auc:.3}");
}
