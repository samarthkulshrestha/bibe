//! Attention validation via a sequence-copy overfit.
//!
//! A single sequence of distinct random token vectors is fed through a
//! self-attention layer that must reconstruct it (target == input, MSE loss).
//! Because every position holds a distinct token, a *uniform* attention map
//! would average all tokens together and could never reconstruct them — so
//! driving the loss to ~0 is only possible if attention learns to concentrate
//! on the matching position. This validates, end to end, that:
//!
//!   * gradients flow through multi-head attention into the Adam optimizer,
//!   * the layer can overfit (loss collapses toward zero),
//!   * the learned attention is non-uniform (not collapsed), and
//!   * every attention row remains a valid distribution (sums to 1).

use bibe::attention::MultiHeadAttention;
use bibe::autograd::Var;
use bibe::optim::Adam;
use bibe::tensor::Tensor;

const SEQ_LEN: usize = 8;
const D_MODEL: usize = 32;
const NUM_HEADS: usize = 4;
const STEPS: usize = 600;
const LR: f32 = 5e-3;

fn main() {
    // Fixed input: one batch of SEQ_LEN distinct random token vectors.
    let x = Var::new(Tensor::randn(&[1, SEQ_LEN, D_MODEL]), false);
    let n_elem = (SEQ_LEN * D_MODEL) as f32;

    let mha = MultiHeadAttention::new(D_MODEL, NUM_HEADS);
    let mut opt = Adam::new(mha.parameters(), LR);

    let mut first_loss = 0.0;
    let mut last_loss = 0.0;

    println!("training self-attention to copy a {SEQ_LEN}-token sequence (d_model={D_MODEL}, heads={NUM_HEADS})\n");

    for step in 0..STEPS {
        opt.zero_grad();

        let (out, _attn) = mha.forward(&x, &x, &x, None);
        let diff = out.sub(&x);
        let loss = diff.mul(&diff).sum().mul_scalar(1.0 / n_elem);

        loss.backward();
        opt.step();

        let l = loss.tensor().data[0];
        if step == 0 {
            first_loss = l;
        }
        last_loss = l;

        if step % 50 == 0 || step == STEPS - 1 {
            println!("  step {step:>4}  mse = {l:.6}");
        }
    }

    // Inspect the final attention map (averaged over heads).
    let (_out, attn) = mha.forward(&x, &x, &x, None);
    let attn_t = attn.tensor();
    let avg = average_heads(&attn_t);

    println!("\nattention map (rows = query position, averaged over heads):");
    print_heatmap(&avg);

    let stats = analyse(&avg);
    let uniform_entropy = (SEQ_LEN as f32).ln();

    println!("\nvalidation:");
    println!("  initial mse ............. {first_loss:.6}");
    println!("  final mse ............... {last_loss:.6}");
    println!("  max |row sum - 1| ....... {:.2e}", stats.row_sum_err);
    println!("  mean row entropy ........ {:.4}  (uniform would be {uniform_entropy:.4})", stats.mean_entropy);
    println!("  peak/uniform ratio ...... {:.2}x", stats.peak_uniform_ratio);

    // --- self-checks -------------------------------------------------------
    // 1. The layer overfit: a single self-attention block reconstructs the
    //    sequence. Uniform attention cannot do this (every output row would be
    //    the mean of all tokens), so collapse here implies discriminative,
    //    non-uniform attention.
    assert!(
        last_loss < first_loss * 0.05,
        "copy loss did not collapse: {first_loss:.6} -> {last_loss:.6}"
    );
    // 2. Each query's attention is a valid probability distribution.
    assert!(
        stats.row_sum_err < 1e-4,
        "attention rows must sum to 1, max error was {:.2e}",
        stats.row_sum_err
    );
    // 3. Attention is non-uniform (not collapsed), confirmed directly on the
    //    learned weights: rows are concentrated relative to the uniform map.
    assert!(
        stats.mean_entropy < 0.93 * uniform_entropy,
        "attention looks near-uniform: entropy {:.4} vs uniform {uniform_entropy:.4}",
        stats.mean_entropy
    );
    assert!(
        stats.peak_uniform_ratio > 1.5,
        "attention not peaked: peak/uniform ratio only {:.2}x",
        stats.peak_uniform_ratio
    );

    println!("\nall attention validation checks passed.");
}

/// Average a `[batch*heads, seq, seq]` attention tensor over heads into
/// `[seq, seq]`. Assumes a single batch element (batch == 1).
fn average_heads(attn: &Tensor) -> Vec<Vec<f32>> {
    let shape = attn.shape();
    let (heads, seq) = (shape[0], shape[1]);
    let mut avg = vec![vec![0.0f32; seq]; seq];
    for h in 0..heads {
        for (i, avg_row) in avg.iter_mut().enumerate() {
            for (j, cell) in avg_row.iter_mut().enumerate() {
                *cell += attn.get(&[h, i, j]) / heads as f32;
            }
        }
    }
    avg
}

struct Stats {
    /// Largest deviation of any attention row from summing to 1.
    row_sum_err: f32,
    /// Mean per-row Shannon entropy (nats); low means concentrated.
    mean_entropy: f32,
    /// Largest single attention weight relative to the uniform value 1/seq.
    peak_uniform_ratio: f32,
}

fn analyse(avg: &[Vec<f32>]) -> Stats {
    let seq = avg.len();
    let uniform = 1.0 / seq as f32;
    let mut row_sum_err = 0.0f32;
    let mut entropy_sum = 0.0f32;
    let mut max_peak = 0.0f32;

    for row in avg {
        let sum: f32 = row.iter().sum();
        row_sum_err = row_sum_err.max((sum - 1.0).abs());

        let entropy: f32 = row
            .iter()
            .filter(|&&p| p > 0.0)
            .map(|&p| -p * p.ln())
            .sum();
        entropy_sum += entropy;

        let peak = row.iter().cloned().fold(0.0f32, f32::max);
        max_peak = max_peak.max(peak);
    }

    Stats {
        row_sum_err,
        mean_entropy: entropy_sum / seq as f32,
        peak_uniform_ratio: max_peak / uniform,
    }
}

fn print_heatmap(avg: &[Vec<f32>]) {
    let ramp = [' ', '.', ':', '-', '=', '+', '*', '#', '%', '@'];
    for row in avg {
        print!("  ");
        for &w in row {
            let idx = ((w * (ramp.len() - 1) as f32).round() as usize).min(ramp.len() - 1);
            print!("{}", ramp[idx]);
        }
        println!();
    }
}
