use crate::autograd::Var;
use crate::nn::LayerNorm;

use super::block::TransformerBlock;

/// A stack of pre-layernorm transformer blocks followed by a final
/// normalization.
///
/// The final LayerNorm is standard for pre-LN architectures: because each
/// block leaves its residual stream un-normalized, a closing norm gives the
/// encoder output a stable scale.
pub struct TransformerEncoder {
    layers: Vec<TransformerBlock>,
    final_norm: LayerNorm,
}

impl TransformerEncoder {
    /// Build an encoder of `num_layers` identical blocks.
    pub fn new(
        num_layers: usize,
        d_model: usize,
        num_heads: usize,
        d_ff: usize,
        dropout_p: f32,
    ) -> Self {
        let layers = (0..num_layers)
            .map(|_| TransformerBlock::new(d_model, num_heads, d_ff, dropout_p))
            .collect();
        TransformerEncoder {
            layers,
            final_norm: LayerNorm::new(d_model),
        }
    }

    /// Forward pass through every block. Returns the normalized output and the
    /// self-attention weights from each layer (in layer order).
    pub fn forward(&self, x: &Var, training: bool) -> (Var, Vec<Var>) {
        let mut hidden = x.clone();
        let mut attn_weights = Vec::with_capacity(self.layers.len());

        for layer in &self.layers {
            let (out, attn) = layer.forward(&hidden, training);
            hidden = out;
            attn_weights.push(attn);
        }

        (self.final_norm.forward(&hidden), attn_weights)
    }

    /// Collect all trainable parameters across every block and the final norm.
    pub fn parameters(&self) -> Vec<Var> {
        let mut params = Vec::new();
        for layer in &self.layers {
            params.extend(layer.parameters());
        }
        params.extend(self.final_norm.parameters());
        params
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::Tensor;

    fn row_stats(t: &Tensor, rows: usize, cols: usize) -> Vec<(f32, f32)> {
        let mut out = Vec::new();
        for r in 0..rows {
            let slice = &t.data[r * cols..(r + 1) * cols];
            let mean: f32 = slice.iter().sum::<f32>() / cols as f32;
            let var: f32 = slice.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / cols as f32;
            out.push((mean, var));
        }
        out
    }

    #[test]
    fn test_output_shape_preserved() {
        let enc = TransformerEncoder::new(4, 32, 4, 64, 0.0);
        let x = Var::new(Tensor::randn(&[2, 6, 32]), false);
        let (y, _attn) = enc.forward(&x, false);
        assert_eq!(y.tensor().shape(), &[2, 6, 32]);
    }

    #[test]
    fn test_collects_attention_per_layer() {
        let enc = TransformerEncoder::new(4, 32, 4, 64, 0.0);
        let x = Var::new(Tensor::randn(&[2, 6, 32]), false);
        let (_y, attn) = enc.forward(&x, false);
        assert_eq!(attn.len(), 4, "expected one attention map per layer");
        for a in &attn {
            assert_eq!(a.tensor().shape(), &[8, 6, 6]);
        }
    }

    #[test]
    fn test_final_norm_applied() {
        // With gamma=1, beta=0 the final norm leaves each output row with
        // mean ~0 and variance ~1.
        let enc = TransformerEncoder::new(3, 16, 2, 32, 0.0);
        let x = Var::new(Tensor::randn(&[2, 5, 16]), false);
        let (y, _attn) = enc.forward(&x, false);
        for (mean, var) in row_stats(&y.tensor(), 2 * 5, 16) {
            assert!(mean.abs() < 1e-3, "row mean not ~0: {mean}");
            assert!((var - 1.0).abs() < 1e-2, "row var not ~1: {var}");
        }
    }

    #[test]
    fn test_output_is_finite() {
        let enc = TransformerEncoder::new(4, 16, 2, 32, 0.0);
        let x = Var::new(Tensor::randn(&[2, 5, 16]), false);
        let (y, _attn) = enc.forward(&x, false);
        assert!(y.tensor().data.iter().all(|v| v.is_finite()), "output has NaN/Inf");
    }

    #[test]
    fn test_deep_stack_gradients_reach_all_layers() {
        // A 4-layer stack must propagate gradients to the very first block's
        // parameters without vanishing to nothing.
        let enc = TransformerEncoder::new(4, 16, 2, 32, 0.0);
        let x = Var::new(Tensor::randn(&[2, 4, 16]), true);
        let (y, _attn) = enc.forward(&x, true);
        let loss = y.sum();
        loss.backward();

        assert!(x.grad().is_some(), "input has no gradient");
        for (i, p) in enc.parameters().iter().enumerate() {
            let g = p.grad().expect("parameter has no gradient");
            assert!(
                g.data.iter().all(|v| v.is_finite()),
                "parameter {i} has non-finite gradient"
            );
        }
    }
}
