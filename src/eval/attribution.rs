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

#[cfg(test)]
mod tests {
    use super::*;

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
