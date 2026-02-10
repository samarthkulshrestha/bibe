use crate::autograd::Var;
use crate::nn::Linear;
use crate::attention::scaled_dot::scaled_dot_product_attention;

/// Multi-head attention mechanism.
///
/// Projects input into multiple heads, applies scaled dot-product attention
/// independently per head, concatenates, and projects back.
///
/// Input: [batch, seq_len, d_model]
/// Output: [batch, seq_len, d_model], attention_weights: [batch, num_heads, seq_len, seq_len]
pub struct MultiHeadAttention {
    pub num_heads: usize,
    pub d_model: usize,
    pub d_k: usize,
    pub w_q: Linear,
    pub w_k: Linear,
    pub w_v: Linear,
    pub w_o: Linear,
}

impl MultiHeadAttention {
    /// Create a new multi-head attention layer.
    ///
    /// `d_model` must be divisible by `num_heads`.
    pub fn new(d_model: usize, num_heads: usize) -> Self {
        assert_eq!(
            d_model % num_heads, 0,
            "d_model ({}) must be divisible by num_heads ({})",
            d_model, num_heads
        );
        let d_k = d_model / num_heads;

        MultiHeadAttention {
            num_heads,
            d_model,
            d_k,
            w_q: Linear::new(d_model, d_model, false),
            w_k: Linear::new(d_model, d_model, false),
            w_v: Linear::new(d_model, d_model, false),
            w_o: Linear::new(d_model, d_model, false),
        }
    }

    /// Forward pass.
    ///
    /// # Arguments
    /// - `query`: [batch, seq_q, d_model]
    /// - `key`:   [batch, seq_k, d_model]
    /// - `value`: [batch, seq_k, d_model]
    /// - `mask`:  Optional [batch, seq_q, seq_k] (0.0 = attend, -1e9 = block)
    ///
    /// For self-attention, pass the same tensor for query, key, value.
    ///
    /// # Returns
    /// `(output, attn_weights)` where:
    /// - `output`:       [batch, seq_q, d_model]
    /// - `attn_weights`: [batch * num_heads, seq_q, seq_k]
    pub fn forward(
        &self,
        query: &Var,
        key: &Var,
        value: &Var,
        mask: Option<&Var>,
    ) -> (Var, Var) {
        let q_shape = query.tensor().shape().to_vec();
        let batch = q_shape[0];

        // Project: [batch, seq, d_model] -> [batch, seq, d_model]
        let q = self.w_q.forward(query);
        let k = self.w_k.forward(key);
        let v = self.w_v.forward(value);

        // Split heads: [batch, seq, d_model] -> [batch*num_heads, seq, d_k]
        let q = q.split_heads(self.num_heads);
        let k = k.split_heads(self.num_heads);
        let v = v.split_heads(self.num_heads);

        // Expand mask for heads if provided:
        // [batch, seq_q, seq_k] -> [batch*num_heads, seq_q, seq_k]
        let mask_expanded = mask.map(|m| m.repeat_batch(self.num_heads));

        // Scaled dot-product attention per head
        let (attn_out, attn_weights) = scaled_dot_product_attention(
            &q, &k, &v,
            mask_expanded.as_ref(),
        );
        // attn_out: [batch*num_heads, seq, d_k]
        // attn_weights: [batch*num_heads, seq_q, seq_k]

        // Merge heads: [batch*num_heads, seq, d_k] -> [batch, seq, d_model]
        let concat = attn_out.merge_heads(batch, self.num_heads);

        // Output projection: [batch, seq, d_model] -> [batch, seq, d_model]
        let output = self.w_o.forward(&concat);

        (output, attn_weights)
    }

    /// Collect all trainable parameters.
    pub fn parameters(&self) -> Vec<Var> {
        let mut params = Vec::new();
        params.extend(self.w_q.parameters());
        params.extend(self.w_k.parameters());
        params.extend(self.w_v.parameters());
        params.extend(self.w_o.parameters());
        params
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::Tensor;

    #[test]
    fn test_output_shape() {
        let mha = MultiHeadAttention::new(8, 2);
        let x = Var::new(Tensor::randn(&[2, 5, 8]), false);
        let (out, attn) = mha.forward(&x, &x, &x, None);

        assert_eq!(out.tensor().shape(), &[2, 5, 8]);
        // attn: [batch*num_heads, seq, seq]
        assert_eq!(attn.tensor().shape(), &[4, 5, 5]);
    }

    #[test]
    fn test_output_shape_large() {
        let mha = MultiHeadAttention::new(32, 4);
        let x = Var::new(Tensor::randn(&[3, 10, 32]), false);
        let (out, attn) = mha.forward(&x, &x, &x, None);

        assert_eq!(out.tensor().shape(), &[3, 10, 32]);
        assert_eq!(attn.tensor().shape(), &[12, 10, 10]);
    }

    #[test]
    fn test_all_projections_receive_gradients() {
        let mha = MultiHeadAttention::new(8, 2);
        let x = Var::new(Tensor::randn(&[1, 3, 8]), true);
        let (out, _) = mha.forward(&x, &x, &x, None);
        let loss = out.sum();
        loss.backward();

        // Input gradient
        assert!(x.grad().is_some(), "input has no gradient");
        assert_eq!(x.grad().unwrap().shape(), &[1, 3, 8]);

        // All projection weight gradients
        assert!(mha.w_q.weight.grad().is_some(), "w_q has no gradient");
        assert!(mha.w_k.weight.grad().is_some(), "w_k has no gradient");
        assert!(mha.w_v.weight.grad().is_some(), "w_v has no gradient");
        assert!(mha.w_o.weight.grad().is_some(), "w_o has no gradient");

        assert_eq!(mha.w_q.weight.grad().unwrap().shape(), &[8, 8]);
        assert_eq!(mha.w_o.weight.grad().unwrap().shape(), &[8, 8]);
    }

    #[test]
    fn test_attn_weights_sum_to_one() {
        let mha = MultiHeadAttention::new(8, 2);
        let x = Var::new(Tensor::randn(&[2, 4, 8]), false);
        let (_, attn) = mha.forward(&x, &x, &x, None);
        let at = attn.tensor();

        // attn shape: [batch*heads, seq, seq] = [4, 4, 4]
        for bh in 0..4 {
            for i in 0..4 {
                let row_sum: f32 = (0..4).map(|j| at.get(&[bh, i, j])).sum();
                assert!(
                    (row_sum - 1.0).abs() < 1e-5,
                    "attn row [{}, {}] sums to {} instead of 1.0", bh, i, row_sum
                );
            }
        }
    }

    #[test]
    fn test_cross_attention() {
        // query and key/value have different seq lengths
        let mha = MultiHeadAttention::new(8, 2);
        let q = Var::new(Tensor::randn(&[1, 3, 8]), true);
        let kv = Var::new(Tensor::randn(&[1, 5, 8]), true);
        let (out, attn) = mha.forward(&q, &kv, &kv, None);

        assert_eq!(out.tensor().shape(), &[1, 3, 8]);
        // attn: [batch*heads, seq_q, seq_k]
        assert_eq!(attn.tensor().shape(), &[2, 3, 5]);

        let loss = out.sum();
        loss.backward();
        assert!(q.grad().is_some());
        assert!(kv.grad().is_some());
    }

    #[test]
    fn test_with_mask() {
        let mha = MultiHeadAttention::new(4, 2);
        let x = Var::new(Tensor::randn(&[1, 3, 4]), false);

        // Causal mask: position i can only attend to positions <= i
        // [1, 3, 3] — row i has 0.0 for j<=i, -1e9 for j>i
        let mask_data = vec![
            0.0,  -1e9, -1e9,
            0.0,   0.0, -1e9,
            0.0,   0.0,  0.0,
        ];
        let mask = Var::new(Tensor::new(mask_data, vec![1, 3, 3]), false);

        let (out, attn) = mha.forward(&x, &x, &x, Some(&mask));
        assert_eq!(out.tensor().shape(), &[1, 3, 4]);

        // Check that masked positions have near-zero attention
        let at = attn.tensor();
        for bh in 0..2 {
            // Row 0: can only attend to pos 0
            assert!(at.get(&[bh, 0, 1]) < 0.01);
            assert!(at.get(&[bh, 0, 2]) < 0.01);
            // Row 1: can attend to pos 0, 1 but not 2
            assert!(at.get(&[bh, 1, 2]) < 0.01);
        }
    }

    #[test]
    fn test_parameters_count() {
        let mha = MultiHeadAttention::new(8, 2);
        // 4 weight matrices, no biases
        assert_eq!(mha.parameters().len(), 4);
    }

    #[test]
    fn test_gradients_finite() {
        let mha = MultiHeadAttention::new(8, 2);
        let x = Var::new(Tensor::randn(&[2, 4, 8]), true);
        let (out, _) = mha.forward(&x, &x, &x, None);
        out.sum().backward();

        for p in mha.parameters() {
            let g = p.grad().expect("parameter missing gradient");
            assert!(
                g.data.iter().all(|v| v.is_finite()),
                "non-finite gradient in MHA parameter"
            );
        }
        let xg = x.grad().unwrap();
        assert!(
            xg.data.iter().all(|v| v.is_finite()),
            "non-finite gradient in MHA input"
        );
    }
}
