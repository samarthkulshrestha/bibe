//! Scoring helpers for attention-rollout attribution.
//!
//! Given the rollout attribution tensor `[batch, seq, seq]` produced by the
//! model, these extract how strongly each source event influences a chosen
//! query event, so source events can be ranked as candidate causes.

use crate::tensor::Tensor;

/// Source-influence scores for `query` from a `[batch, seq, seq]` rollout
/// attribution tensor: element `j` is how much source position `j` influences
/// the query position through the whole stack.
pub fn attribution_row(attribution: &Tensor, batch: usize, query: usize) -> Vec<f32> {
    let seq = attribution.shape()[1];
    (0..seq).map(|j| attribution.get(&[batch, query, j])).collect()
}

/// Head-averaged attention for `query` from a single layer's raw attention
/// tensor `[batch*num_heads, seq, seq]`: element `j` is the mean over heads of
/// how much the query attends to source `j`. Unlike [`attribution_row`] on the
/// rollout, this preserves a single layer's sharp attention without residual
/// mixing across layers.
pub fn head_averaged_query_row(
    attn: &Tensor,
    num_heads: usize,
    batch: usize,
    query: usize,
) -> Vec<f32> {
    let seq = attn.shape()[1];
    let mut row = vec![0.0f32; seq];
    for h in 0..num_heads {
        let group = batch * num_heads + h;
        for (j, cell) in row.iter_mut().enumerate() {
            *cell += attn.get(&[group, query, j]);
        }
    }
    let inv = 1.0 / num_heads as f32;
    row.iter_mut().for_each(|v| *v *= inv);
    row
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_head_averaged_query_row() {
        // [2, 3, 3]: 1 batch, 2 heads, seq 3. Query 1 rows:
        //   head 0: [0.1, 0.8, 0.1], head 1: [0.3, 0.4, 0.3] -> avg [0.2, 0.6, 0.2]
        let data = vec![
            0.0, 0.0, 0.0, 0.1, 0.8, 0.1, 0.0, 0.0, 0.0, // head 0
            0.0, 0.0, 0.0, 0.3, 0.4, 0.3, 0.0, 0.0, 0.0, // head 1
        ];
        let attn = Tensor::new(data, vec![2, 3, 3]);
        let row = head_averaged_query_row(&attn, 2, 0, 1);
        assert_eq!(row.len(), 3);
        for (a, e) in row.iter().zip([0.2, 0.6, 0.2]) {
            assert!((a - e).abs() < 1e-6, "{a} != {e}");
        }
    }

    #[test]
    fn test_extracts_query_row() {
        // [1, 3, 3]: rows are [.1 .2 .7], [.5 .4 .1], [.3 .3 .4]
        let a = Tensor::new(
            vec![0.1, 0.2, 0.7, 0.5, 0.4, 0.1, 0.3, 0.3, 0.4],
            vec![1, 3, 3],
        );
        assert_eq!(attribution_row(&a, 0, 1), vec![0.5, 0.4, 0.1]);
        assert_eq!(attribution_row(&a, 0, 2), vec![0.3, 0.3, 0.4]);
    }
}
