//! Out-of-distribution generalization study.
//!
//! Leave-one-out across the four bug kinds: train on three, measure detection
//! AUC on the held-out fourth. Repeated across a sweep of contrastive-loss
//! weights to see whether leaning on the normal-vs-anomalous objective (rather
//! than per-kind focal signatures) improves generalization to unseen bugs.
//!
//! ```text
//! cargo run --release --example ood_study
//! ```

use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;

use bibe::autograd::Var;
use bibe::data::{collate, extract_windows, BugKind, DataLoader, Trace, TraceGenerator, Vocabulary, N_AUX};
use bibe::eval::auc_roc;
use bibe::model::{BibeConfig, BibeModel};
use bibe::train::{TrainConfig, Trainer};

const WINDOW: usize = 64;
const BATCH_SIZE: usize = 8;
const EPOCHS: usize = 20;
const SEED: u64 = 1234;

fn main() {
    println!("leave-one-out OOD detection AUC (train on 3 kinds, test on the held-out 4th)\n");

    // Attribution supervision is left off here so the study isolates detection
    // generalization and the effect of the contrastive weight.
    let lambdas = [0.0, 1.0, 4.0, 8.0];

    println!("(ID = unseen traces of the trained kinds; OOD = the held-out kind)\n");
    let n = BugKind::ALL.len() as f32;
    for &contrastive_lambda in &lambdas {
        println!("contrastive_lambda = {contrastive_lambda}");
        let (mut id_sum, mut ood_sum) = (0.0, 0.0);
        for &held_out in &BugKind::ALL {
            let (id, ood) = leave_one_out(held_out, contrastive_lambda);
            println!("  held-out {:<16} ID {id:.3}   OOD {ood:.3}", format!("{held_out:?}"));
            id_sum += id;
            ood_sum += ood;
        }
        println!("  mean   ID {:.3}   OOD {:.3}\n", id_sum / n, ood_sum / n);
    }
}

/// Train without `held_out`; return (in-distribution AUC on unseen traces of
/// the trained kinds, out-of-distribution AUC on the held-out kind).
fn leave_one_out(held_out: BugKind, contrastive_lambda: f32) -> (f32, f32) {
    let train_kinds: Vec<BugKind> =
        BugKind::ALL.iter().copied().filter(|&k| k != held_out).collect();

    let train_set = TraceGenerator::new(13).dataset_with_kinds(60, 30, &train_kinds);
    let vocab = Vocabulary::build(&train_set, 1);
    let trainer = train_on(&train_set, &vocab, contrastive_lambda);

    // In-distribution: fresh traces of the trained kinds (different seed).
    let id_set = TraceGenerator::new(71).dataset_with_kinds(20, 20, &train_kinds);
    let id = detection_auc(trainer.model(), &vocab, &id_set);
    // Out-of-distribution: the held-out kind.
    let ood_set = TraceGenerator::new(57).dataset_with_kinds(0, 20, &[held_out]);
    let ood = detection_auc(trainer.model(), &vocab, &ood_set);
    (id, ood)
}

fn train_on(dataset: &[Trace], vocab: &Vocabulary, contrastive_lambda: f32) -> Trainer {
    let mut windows = Vec::new();
    for t in dataset {
        windows.extend(extract_windows(t, WINDOW, WINDOW));
    }
    let mut rng = StdRng::seed_from_u64(7);
    windows.shuffle(&mut rng);

    let loader = DataLoader::new(windows, BATCH_SIZE);
    let steps_per_epoch = loader.num_batches();

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
        warmup_steps: steps_per_epoch,
        total_steps: EPOCHS * steps_per_epoch,
        grad_clip: 1.0,
        contrastive_lambda,
        attribution_lambda: 0.0,
        ..TrainConfig::default()
    };
    let mut trainer = Trainer::new(BibeModel::new(&config), train_cfg);
    for _ in 0..EPOCHS {
        trainer.train_epoch(&loader, vocab);
    }
    trainer
}

/// Pooled event-level detection AUC over a held-out set.
fn detection_auc(model: &BibeModel, vocab: &Vocabulary, eval_set: &[Trace]) -> f32 {
    let mut scores = Vec::new();
    let mut labels = Vec::new();
    for trace in eval_set {
        for window in extract_windows(trace, WINDOW, WINDOW) {
            let batch = collate(&[window], vocab);
            let aux = Var::new(batch.aux.clone(), false);
            let out = model.forward(&batch.function_ids, &aux, 1, batch.seq, false);
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
