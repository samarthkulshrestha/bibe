//! Activation function wrappers for neural network layers.
//!
//! These delegate to the corresponding `Var` methods, which are backed
//! by fused backward implementations in `autograd::backward`.

use crate::autograd::Var;

/// GeLU activation: 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
pub fn gelu(x: &Var) -> Var {
    x.gelu()
}

/// ReLU activation: max(0, x)
pub fn relu(x: &Var) -> Var {
    x.relu()
}

/// Sigmoid activation: 1 / (1 + exp(-x))
pub fn sigmoid(x: &Var) -> Var {
    x.sigmoid()
}

/// Softmax along a dimension (uses numerically stable implementation).
pub fn softmax(x: &Var, dim: usize) -> Var {
    x.softmax(dim)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::Tensor;
    use crate::autograd::gradcheck;

    // --- Forward correctness ---

    #[test]
    fn test_relu_forward() {
        let x = Var::new(Tensor::new(vec![-2.0, -1.0, 0.0, 1.0, 2.0], vec![5]), false);
        let y = relu(&x).tensor();
        assert_eq!(y.data, vec![0.0, 0.0, 0.0, 1.0, 2.0]);
    }

    #[test]
    fn test_sigmoid_forward() {
        let x = Var::new(Tensor::new(vec![0.0], vec![1]), false);
        let y = sigmoid(&x).tensor();
        assert!((y.data[0] - 0.5).abs() < 1e-6);

        // Large positive -> ~1
        let x2 = Var::new(Tensor::new(vec![100.0], vec![1]), false);
        let y2 = sigmoid(&x2).tensor();
        assert!((y2.data[0] - 1.0).abs() < 1e-5);

        // Large negative -> ~0
        let x3 = Var::new(Tensor::new(vec![-100.0], vec![1]), false);
        let y3 = sigmoid(&x3).tensor();
        assert!(y3.data[0].abs() < 1e-5);
    }

    #[test]
    fn test_gelu_forward() {
        // GeLU(0) = 0
        let x = Var::new(Tensor::new(vec![0.0], vec![1]), false);
        let y = gelu(&x).tensor();
        assert!(y.data[0].abs() < 1e-6);

        // GeLU is approximately x for large positive x
        let x2 = Var::new(Tensor::new(vec![3.0], vec![1]), false);
        let y2 = gelu(&x2).tensor();
        assert!((y2.data[0] - 3.0).abs() < 0.01);

        // GeLU is approximately 0 for large negative x
        let x3 = Var::new(Tensor::new(vec![-3.0], vec![1]), false);
        let y3 = gelu(&x3).tensor();
        assert!(y3.data[0].abs() < 0.01);
    }

    #[test]
    fn test_gelu_matches_pytorch() {
        // PyTorch reference values for GeLU at specific points
        // gelu(-1.0) ≈ -0.1588
        // gelu(1.0) ≈ 0.8412
        let x = Var::new(Tensor::new(vec![-1.0, 1.0], vec![2]), false);
        let y = gelu(&x).tensor();
        assert!((y.data[0] - (-0.1588)).abs() < 0.001, "gelu(-1) = {}", y.data[0]);
        assert!((y.data[1] - 0.8412).abs() < 0.001, "gelu(1) = {}", y.data[1]);
    }

    // --- Gradient checking ---

    #[test]
    fn gradcheck_relu() {
        // Avoid x=0 where ReLU is non-differentiable
        let x = Tensor::new(vec![-2.0, -0.5, 0.5, 1.0, 3.0], vec![5]);
        let (ok, err) = gradcheck(&|v| v.relu().sum(), &x, 5e-4, 1e-2);
        assert!(ok, "relu gradcheck failed: max_rel_err={}", err);
    }

    #[test]
    fn gradcheck_sigmoid() {
        let x = Tensor::new(vec![-2.0, -1.0, 0.0, 1.0, 2.0], vec![5]);
        let (ok, err) = gradcheck(&|v| v.sigmoid().sum(), &x, 5e-4, 1e-2);
        assert!(ok, "sigmoid gradcheck failed: max_rel_err={}", err);
    }

    #[test]
    fn gradcheck_tanh() {
        let x = Tensor::new(vec![-2.0, -1.0, 0.0, 1.0, 2.0], vec![5]);
        let (ok, err) = gradcheck(&|v| v.tanh().sum(), &x, 5e-4, 1e-2);
        assert!(ok, "tanh gradcheck failed: max_rel_err={}", err);
    }

    #[test]
    fn gradcheck_gelu() {
        let x = Tensor::new(vec![-2.0, -1.0, 0.0, 1.0, 2.0], vec![5]);
        let (ok, err) = gradcheck(&|v| v.gelu().sum(), &x, 5e-4, 1e-2);
        assert!(ok, "gelu gradcheck failed: max_rel_err={}", err);
    }

    #[test]
    fn gradcheck_gelu_weighted() {
        // Non-trivial gradient through gelu
        let x = Tensor::new(vec![0.5, 1.0, 1.5, -0.5, -1.0, -1.5], vec![2, 3]);
        let (ok, err) = gradcheck(
            &|v| {
                let g = v.gelu();
                let w = Var::new(
                    Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]),
                    false,
                );
                g.mul(&w).sum()
            },
            &x, 5e-4, 1e-2,
        );
        assert!(ok, "gelu weighted gradcheck failed: max_rel_err={}", err);
    }

    #[test]
    fn gradcheck_sigmoid_chain() {
        // sigmoid into linear combination
        let x = Tensor::new(vec![-1.0, 0.0, 1.0, 2.0], vec![2, 2]);
        let (ok, err) = gradcheck(
            &|v| {
                let s = v.sigmoid();
                let w = Var::new(
                    Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]),
                    false,
                );
                s.mul(&w).sum()
            },
            &x, 5e-4, 1e-2,
        );
        assert!(ok, "sigmoid chain gradcheck failed: max_rel_err={}", err);
    }
}
