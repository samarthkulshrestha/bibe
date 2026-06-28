//! End-to-end training run on synthetic data.
//!
//! Generates a labeled synthetic corpus, builds a vocabulary, windows and
//! batches it, then trains the BiBE model (focal + attention-sparsity +
//! contrastive losses, warmup/cosine LR, gradient clipping, Adam) and writes a
//! checkpoint. Run with:
//!
//! ```text
//! cargo run --release --example train
//! ```

use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;

use bibe::autograd::Var;
use bibe::data::{collate, extract_windows, DataLoader, TraceGenerator, Vocabulary, N_AUX};
use bibe::eval::{auc_roc, hit_at_k, mrr, precision_at_k, rank_by_score_desc};
use bibe::model::{BibeConfig, BibeModel};
use bibe::train::{save_parameters, TrainConfig, Trainer};

const WINDOW: usize = 64;
const BATCH_SIZE: usize = 8;
const EPOCHS: usize = 20;
const SEED: u64 = 1234;

fn main() {
    // 1. Generate a deterministic labeled dataset.
    let mut generator = TraceGenerator::new(42);
    let dataset = generator.dataset(60, 30); // 60 normal, 30 anomalous
    let n_anom = dataset.iter().filter(|t| t.is_anomalous()).count();
    println!("generated {} traces ({n_anom} anomalous)", dataset.len());

    // 2. Build the function vocabulary.
    let vocab = Vocabulary::build(&dataset, 1);
    println!("vocabulary size: {}", vocab.len());

    // 3. Window the traces, then shuffle so each batch mixes normal and
    //    anomalous windows (the contrastive loss needs both in a batch).
    let mut windows = Vec::new();
    for t in &dataset {
        windows.extend(extract_windows(t, WINDOW, WINDOW));
    }
    let mut rng = StdRng::seed_from_u64(7);
    windows.shuffle(&mut rng);
    println!("extracted {} windows", windows.len());

    let loader = DataLoader::new(windows, BATCH_SIZE);
    let steps_per_epoch = loader.num_batches();

    // 4. Build the model (seed weight init for reproducible runs).
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
    let model = BibeModel::new(&config);

    // 5. Train.
    let train_cfg = TrainConfig {
        lr: 1e-3,
        warmup_steps: steps_per_epoch,
        total_steps: EPOCHS * steps_per_epoch,
        grad_clip: 1.0,
        ..TrainConfig::default()
    };
    let mut trainer = Trainer::new(model, train_cfg);

    println!("\ntraining {EPOCHS} epochs x {steps_per_epoch} batches:");
    let mut first = None;
    let mut last = 0.0;
    for epoch in 0..EPOCHS {
        let loss = trainer.train_epoch(&loader, &vocab);
        if first.is_none() {
            first = Some(loss);
        }
        last = loss;
        println!("  epoch {epoch:>2}  mean loss {loss:.4}");
    }
    let first = first.unwrap();
    println!("\nloss: {first:.4} -> {last:.4}  ({:.0}% reduction)", 100.0 * (1.0 - last / first));

    // 6. Evaluate on a held-out set with a different seed.
    let eval_set = TraceGenerator::new(99).dataset(20, 20);
    evaluate(trainer.model(), &vocab, &eval_set);

    // 7. Checkpoint.
    let path = std::path::Path::new("bibe_checkpoint.bin");
    save_parameters(path, &trainer.model().parameters()).expect("failed to save checkpoint");
    println!("saved checkpoint to {}", path.display());
}

/// Run the model over held-out traces and report detection and localization
/// metrics against the generator's known root causes.
fn evaluate(model: &BibeModel, vocab: &Vocabulary, eval_set: &[bibe::data::Trace]) {
    // Pooled per-event scores/labels for detection metrics.
    let mut all_scores = Vec::new();
    let mut all_labels = Vec::new();
    // Per-anomalous-window localization tallies.
    let (mut hits1, mut hits5, mut mrr_sum, mut n_anom) = (0usize, 0usize, 0.0f32, 0usize);

    for trace in eval_set {
        for window in extract_windows(trace, WINDOW, WINDOW) {
            let batch = collate(&[window], vocab);
            let aux = Var::new(batch.aux.clone(), false);
            let out = model.forward(&batch.function_ids, &aux, 1, batch.seq, false);
            let scores = out.anomaly_scores.tensor().data;

            // Restrict to real (non-padded) positions.
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
                if hit_at_k(&ranked, rc, 1) {
                    hits1 += 1;
                }
                if hit_at_k(&ranked, rc, 5) {
                    hits5 += 1;
                }
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
