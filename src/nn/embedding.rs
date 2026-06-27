use crate::autograd::Var;
use crate::autograd::backward::EmbeddingBackward;
use crate::tensor::Tensor;

/// Learned embedding lookup table.
///
/// Holds a `[vocab_size, d_model]` weight matrix; a forward pass gathers the
/// rows named by a set of integer indices. The backward pass scatters
/// gradients back into the table, accumulating when an index appears more
/// than once.
pub struct Embedding {
    pub weight: Var,
    d_model: usize,
}

impl Embedding {
    /// Create an embedding table with the given vocabulary size and
    /// dimensionality, initialized from a standard normal.
    pub fn new(vocab_size: usize, d_model: usize) -> Self {
        Embedding {
            weight: Var::new(Tensor::randn(&[vocab_size, d_model]), true),
            d_model,
        }
    }

    /// Look up `indices` (a flat list addressing `leading_shape` positions) and
    /// return embeddings of shape `[leading_shape..., d_model]`.
    pub fn forward(&self, indices: &[usize], leading_shape: &[usize]) -> Var {
        let expected: usize = leading_shape.iter().product();
        assert_eq!(
            indices.len(), expected,
            "index count {} does not match leading shape {:?}",
            indices.len(), leading_shape
        );

        let weight = self.weight.tensor();
        let vocab_size = weight.shape()[0];
        let d = self.d_model;

        // Gather: copy each indexed weight row into the output.
        let mut data = Vec::with_capacity(indices.len() * d);
        for &idx in indices {
            assert!(idx < vocab_size, "index {idx} out of vocab range {vocab_size}");
            data.extend_from_slice(&weight.data[idx * d..(idx + 1) * d]);
        }

        let mut out_shape = leading_shape.to_vec();
        out_shape.push(d);

        Var::from_op(
            Tensor::new(data, out_shape),
            Box::new(EmbeddingBackward {
                indices: indices.to_vec(),
                vocab_size,
                d_model: d,
            }),
            vec![self.weight.clone()],
        )
    }

    /// Collect all trainable parameters.
    pub fn parameters(&self) -> Vec<Var> {
        vec![self.weight.clone()]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::autograd::gradcheck::gradcheck;

    #[test]
    fn test_gathers_correct_rows() {
        // weight rows: 0 -> [1,2], 1 -> [3,4], 2 -> [5,6]
        let emb = Embedding::new(3, 2);
        emb.weight.with_data_mut(|t| {
            t.data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        });
        let out = emb.forward(&[2, 0], &[2]);
        assert_eq!(out.tensor().shape(), &[2, 2]);
        assert_eq!(out.tensor().data, vec![5.0, 6.0, 1.0, 2.0]);
    }

    #[test]
    fn test_output_shape_2d_indices() {
        let emb = Embedding::new(10, 4);
        // batch=2, seq=3 -> [2, 3, 4]
        let out = emb.forward(&[0, 1, 2, 3, 4, 5], &[2, 3]);
        assert_eq!(out.tensor().shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_repeated_index_accumulates_gradient() {
        // Index 0 used twice, index 1 once, index 2 never.
        let emb = Embedding::new(3, 2);
        let out = emb.forward(&[0, 0, 1], &[3]);
        let loss = out.sum();
        loss.backward();

        let g = emb.weight.grad().unwrap();
        assert_eq!(g.shape(), &[3, 2]);
        // Row 0 receives gradient from two positions, row 1 from one, row 2 none.
        assert_eq!(g.data, vec![2.0, 2.0, 1.0, 1.0, 0.0, 0.0]);
    }

    #[test]
    fn test_weight_gradient_numeric() {
        let w = Tensor::new(vec![0.5, -1.0, 2.0, 3.0, 0.0, 1.5], vec![3, 2]);
        let indices = [0, 2, 2, 1];
        let (ok, err) = gradcheck(
            &|weight: &Var| {
                let emb = Embedding { weight: weight.clone(), d_model: 2 };
                emb.forward(&indices, &[4]).pow(2.0).sum()
            },
            &w,
            1e-3,
            2e-3,
        );
        assert!(ok, "embedding weight gradcheck failed, max rel err = {err}");
    }
}
