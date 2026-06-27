use crate::tensor::Tensor;
use crate::tensor::matmul::batched_matmul;

/// Attention rollout for causal attribution (Abnar & Zuidema, 2020).
///
/// Given the per-layer self-attention weights `[batch * num_heads, seq, seq]`,
/// this:
///   1. averages attention over heads,
///   2. mixes in the residual connection as `0.5·A + 0.5·I` and row-normalizes,
///   3. multiplies the resulting row-stochastic matrices across layers.
///
/// The result `[batch, seq, seq]` approximates how much each source position
/// influences each query position once information has flowed through the
/// whole stack. Rows sum to 1. This is an inference-time analysis utility, so
/// it operates on raw tensor values rather than through the autograd graph.
pub fn attention_rollout(layer_attn: &[Tensor], num_heads: usize) -> Tensor {
    assert!(!layer_attn.is_empty(), "need at least one attention layer");

    let seq = layer_attn[0].shape()[1];
    let batch = layer_attn[0].shape()[0] / num_heads;

    let mut rollout: Option<Tensor> = None;
    for attn in layer_attn {
        let aug = augmented_head_average(attn, batch, num_heads, seq);
        rollout = Some(match rollout {
            // Later layers multiply on the left of the accumulated product.
            Some(acc) => batched_matmul(&aug, &acc),
            None => aug,
        });
    }

    rollout.unwrap()
}

/// Average over heads, mix in the residual as `0.5·A + 0.5·I`, and
/// row-normalize. Returns `[batch, seq, seq]`.
fn augmented_head_average(attn: &Tensor, batch: usize, num_heads: usize, seq: usize) -> Tensor {
    let mut data = vec![0.0f32; batch * seq * seq];

    for b in 0..batch {
        for i in 0..seq {
            // Head-averaged, residual-mixed row.
            let mut row = vec![0.0f32; seq];
            for (j, cell) in row.iter_mut().enumerate() {
                let mut avg = 0.0;
                for h in 0..num_heads {
                    avg += attn.get(&[b * num_heads + h, i, j]);
                }
                avg /= num_heads as f32;
                *cell = 0.5 * avg + if i == j { 0.5 } else { 0.0 };
            }

            // Row-normalize so the matrix stays row-stochastic.
            let sum: f32 = row.iter().sum();
            let inv = if sum > 0.0 { 1.0 / sum } else { 0.0 };
            for (j, &v) in row.iter().enumerate() {
                data[b * seq * seq + i * seq + j] = v * inv;
            }
        }
    }

    Tensor::new(data, vec![batch, seq, seq])
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a `[batch*heads, seq, seq]` attention tensor where every head is
    /// the identity (each position attends only to itself).
    fn identity_attn(batch: usize, heads: usize, seq: usize) -> Tensor {
        let mut data = vec![0.0f32; batch * heads * seq * seq];
        for bh in 0..batch * heads {
            for i in 0..seq {
                data[bh * seq * seq + i * seq + i] = 1.0;
            }
        }
        Tensor::new(data, vec![batch * heads, seq, seq])
    }

    fn uniform_attn(batch: usize, heads: usize, seq: usize) -> Tensor {
        let v = 1.0 / seq as f32;
        Tensor::new(vec![v; batch * heads * seq * seq], vec![batch * heads, seq, seq])
    }

    #[test]
    fn test_output_shape() {
        let attn = vec![uniform_attn(2, 4, 5), uniform_attn(2, 4, 5)];
        let out = attention_rollout(&attn, 4);
        assert_eq!(out.shape(), &[2, 5, 5]);
    }

    #[test]
    fn test_rows_sum_to_one() {
        let attn = vec![uniform_attn(1, 2, 4), uniform_attn(1, 2, 4), uniform_attn(1, 2, 4)];
        let out = attention_rollout(&attn, 2);
        let seq = 4;
        for i in 0..seq {
            let sum: f32 = (0..seq).map(|j| out.get(&[0, i, j])).sum();
            assert!((sum - 1.0).abs() < 1e-5, "row {i} sums to {sum}");
        }
    }

    #[test]
    fn test_identity_attention_rolls_out_to_identity() {
        // Identity attention at every layer -> 0.5I + 0.5I = I -> product is I.
        let attn = vec![identity_attn(1, 2, 3), identity_attn(1, 2, 3)];
        let out = attention_rollout(&attn, 2);
        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (out.get(&[0, i, j]) - expected).abs() < 1e-5,
                    "rollout[{i},{j}] = {} expected {expected}",
                    out.get(&[0, i, j])
                );
            }
        }
    }

    #[test]
    fn test_single_layer_is_residual_mix() {
        // One layer of identity attention -> 0.5*I + 0.5*I = I after normalize.
        let attn = vec![identity_attn(1, 1, 2)];
        let out = attention_rollout(&attn, 1);
        assert!((out.get(&[0, 0, 0]) - 1.0).abs() < 1e-5);
        assert!((out.get(&[0, 0, 1]) - 0.0).abs() < 1e-5);
    }
}
