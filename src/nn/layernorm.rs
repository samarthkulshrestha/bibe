use crate::autograd::Var;
use crate::tensor::Tensor;

/// Layer normalization over the last dimension.
///
/// ```text
/// y = γ ⊙ (x - μ) / √(σ² + ε) + β
/// ```
///
/// where μ and σ² are the mean and population variance computed across the
/// final (feature) dimension. `gamma` and `beta` are learned per-feature
/// scale and shift parameters of shape `[normalized_size]`.
pub struct LayerNorm {
    pub gamma: Var,
    pub beta: Var,
    eps: f32,
}

impl LayerNorm {
    /// Create a LayerNorm over `normalized_size` features with the standard
    /// epsilon (1e-5). `gamma` starts at 1 and `beta` at 0, so the layer is
    /// initially a pure normalization.
    pub fn new(normalized_size: usize) -> Self {
        Self::with_eps(normalized_size, 1e-5)
    }

    /// Create a LayerNorm with an explicit epsilon.
    pub fn with_eps(normalized_size: usize, eps: f32) -> Self {
        LayerNorm {
            gamma: Var::new(Tensor::ones(&[normalized_size]), true),
            beta: Var::new(Tensor::zeros(&[normalized_size]), true),
            eps,
        }
    }

    /// Normalize `x` across its last dimension and apply the affine transform.
    pub fn forward(&self, x: &Var) -> Var {
        let last = x.tensor().shape().len() - 1;

        let mean = x.mean(last);
        let var = x.var(last);
        let std = var.add_scalar(self.eps).sqrt();

        let normed = x.sub(&mean).div(&std);
        // gamma/beta are [features]; they broadcast over the leading dims.
        normed.mul(&self.gamma).add(&self.beta)
    }

    /// Collect all trainable parameters.
    pub fn parameters(&self) -> Vec<Var> {
        vec![self.gamma.clone(), self.beta.clone()]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::autograd::gradcheck::gradcheck;

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
    fn test_preserves_shape() {
        let ln = LayerNorm::new(4);
        let x = Var::new(Tensor::randn(&[3, 4]), false);
        let y = ln.forward(&x);
        assert_eq!(y.tensor().shape(), &[3, 4]);
    }

    #[test]
    fn test_normalizes_each_row() {
        // With gamma=1, beta=0 the output rows have mean ~0 and variance ~1.
        let ln = LayerNorm::new(5);
        let x = Var::new(
            Tensor::new(
                vec![10.0, 12.0, 14.0, 16.0, 18.0, -3.0, -1.0, 0.0, 5.0, 9.0],
                vec![2, 5],
            ),
            false,
        );
        let y = ln.forward(&x);
        for (mean, var) in row_stats(&y.tensor(), 2, 5) {
            assert!(mean.abs() < 1e-3, "row mean not ~0: {mean}");
            assert!((var - 1.0).abs() < 1e-2, "row var not ~1: {var}");
        }
    }

    #[test]
    fn test_affine_parameters_applied() {
        // Set gamma=2, beta=1; normalized output should scale and shift.
        let ln = LayerNorm::new(4);
        ln.gamma.with_data_mut(|t| t.data.iter_mut().for_each(|v| *v = 2.0));
        ln.beta.with_data_mut(|t| t.data.iter_mut().for_each(|v| *v = 1.0));
        let x = Var::new(Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]), false);
        let y = ln.forward(&x);
        // Each output row still has mean = beta and variance = gamma^2.
        let (mean, var) = row_stats(&y.tensor(), 1, 4)[0];
        assert!((mean - 1.0).abs() < 1e-3, "mean should equal beta: {mean}");
        assert!((var - 4.0).abs() < 1e-2, "var should equal gamma^2: {var}");
    }

    #[test]
    fn test_parameters_receive_gradients() {
        let ln = LayerNorm::new(4);
        let x = Var::new(Tensor::randn(&[2, 4]), true);
        let loss = ln.forward(&x).sum();
        loss.backward();
        assert!(ln.gamma.grad().is_some(), "gamma has no gradient");
        assert!(ln.beta.grad().is_some(), "beta has no gradient");
        assert!(x.grad().is_some(), "input has no gradient");
    }

    #[test]
    fn test_input_gradient_numeric() {
        // Use a fixed non-uniform weighted sum as the loss. Plain sum (or
        // sum-of-squares) is invariant to x because each normalized row has
        // fixed mean 0 and variance 1, which would make the gradient
        // degenerate; weighting breaks that symmetry.
        let x = Tensor::new(vec![0.5, -1.0, 2.0, 3.0, 0.0, 1.5], vec![2, 3]);
        let weights = Var::new(
            Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]),
            false,
        );
        let (ok, err) = gradcheck(
            &|v: &Var| {
                let ln = LayerNorm::new(3);
                ln.forward(v).mul(&weights).sum()
            },
            &x,
            1e-3,
            2e-3,
        );
        assert!(ok, "layernorm input gradcheck failed, max rel err = {err}");
    }
}
