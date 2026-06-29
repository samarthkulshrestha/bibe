use crate::attention::MultiHeadAttention;
use crate::autograd::Var;
use crate::nn::{Dropout, LayerNorm, PositionwiseFFN};

/// A single pre-layernorm transformer encoder block.
///
/// ```text
/// h = x + dropout(attention(norm1(x)))
/// y = h + dropout(ffn(norm2(h)))
/// ```
///
/// Pre-LN (normalizing the input to each sub-layer rather than its output)
/// keeps a clean residual path, which gives more stable gradients when many
/// blocks are stacked. Attention is fully bidirectional (no causal mask).
pub struct TransformerBlock {
    attention: MultiHeadAttention,
    ffn: PositionwiseFFN,
    norm1: LayerNorm,
    norm2: LayerNorm,
    dropout: Dropout,
}

impl TransformerBlock {
    /// Build a block with `num_heads` attention heads, a feed-forward hidden
    /// dimension of `d_ff`, and dropout probability `dropout_p`.
    pub fn new(d_model: usize, num_heads: usize, d_ff: usize, dropout_p: f32) -> Self {
        TransformerBlock {
            attention: MultiHeadAttention::new(d_model, num_heads),
            ffn: PositionwiseFFN::new(d_model, d_ff),
            norm1: LayerNorm::new(d_model),
            norm2: LayerNorm::new(d_model),
            dropout: Dropout::new(dropout_p),
        }
    }

    /// Forward pass. Returns the block output and the attention weights from
    /// the self-attention sub-layer.
    pub fn forward(&self, x: &Var, training: bool, mask: Option<&Var>) -> (Var, Var) {
        // Attention sub-layer (pre-norm + residual).
        let x_norm = self.norm1.forward(x);
        let (attn_out, attn_weights) =
            self.attention.forward(&x_norm, &x_norm, &x_norm, mask);
        let h = x.add(&self.dropout.forward(&attn_out, training));

        // Feed-forward sub-layer (pre-norm + residual).
        let h_norm = self.norm2.forward(&h);
        let ffn_out = self.ffn.forward(&h_norm);
        let y = h.add(&self.dropout.forward(&ffn_out, training));

        (y, attn_weights)
    }

    /// Collect all trainable parameters.
    pub fn parameters(&self) -> Vec<Var> {
        let mut params = self.attention.parameters();
        params.extend(self.ffn.parameters());
        params.extend(self.norm1.parameters());
        params.extend(self.norm2.parameters());
        params
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::Tensor;

    #[test]
    fn test_output_shape_preserved() {
        let block = TransformerBlock::new(32, 4, 64, 0.0);
        let x = Var::new(Tensor::randn(&[2, 6, 32]), false);
        let (y, _attn) = block.forward(&x, false, None);
        assert_eq!(y.tensor().shape(), &[2, 6, 32]);
    }

    #[test]
    fn test_attention_weights_shape() {
        let block = TransformerBlock::new(32, 4, 64, 0.0);
        let x = Var::new(Tensor::randn(&[2, 6, 32]), false);
        let (_y, attn) = block.forward(&x, false, None);
        // [batch * num_heads, seq, seq]
        assert_eq!(attn.tensor().shape(), &[8, 6, 6]);
    }

    #[test]
    fn test_transforms_input() {
        let block = TransformerBlock::new(16, 2, 32, 0.0);
        let x = Var::new(Tensor::randn(&[1, 4, 16]), false);
        let (y, _attn) = block.forward(&x, false, None);
        let differs = x
            .tensor()
            .data
            .iter()
            .zip(y.tensor().data.iter())
            .any(|(a, b)| (a - b).abs() > 1e-5);
        assert!(differs, "block output should differ from its input");
    }

    #[test]
    fn test_output_is_finite() {
        let block = TransformerBlock::new(16, 2, 32, 0.0);
        let x = Var::new(Tensor::randn(&[2, 5, 16]), false);
        let (y, _attn) = block.forward(&x, false, None);
        assert!(y.tensor().data.iter().all(|v| v.is_finite()), "output has NaN/Inf");
    }

    #[test]
    fn test_all_parameters_receive_gradients() {
        let block = TransformerBlock::new(16, 2, 32, 0.0);
        let x = Var::new(Tensor::randn(&[2, 4, 16]), true);
        let (y, _attn) = block.forward(&x, true, None);
        let loss = y.sum();
        loss.backward();

        assert!(x.grad().is_some(), "input has no gradient");
        for (i, p) in block.parameters().iter().enumerate() {
            assert!(p.grad().is_some(), "parameter {i} has no gradient");
        }
    }
}
