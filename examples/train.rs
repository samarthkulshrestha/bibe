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

use bibe::data::{extract_windows, DataLoader, TraceGenerator, Vocabulary, N_AUX};
use bibe::model::{BibeConfig, BibeModel};
use bibe::train::{save_parameters, TrainConfig, Trainer};

const WINDOW: usize = 64;
const BATCH_SIZE: usize = 8;
const EPOCHS: usize = 20;

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

    // 4. Build the model.
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

    // 6. Checkpoint.
    let path = std::path::Path::new("bibe_checkpoint.bin");
    save_parameters(path, &trainer.model().parameters()).expect("failed to save checkpoint");
    println!("saved checkpoint to {}", path.display());
}
