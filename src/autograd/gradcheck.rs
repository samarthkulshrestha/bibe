use crate::tensor::Tensor;
use super::graph::Var;

/// Compute numerical gradient of a scalar-valued function at point `x`
/// using central finite differences: (f(x+eps) - f(x-eps)) / (2*eps).
///
/// `f` takes a `&Tensor` input and returns a scalar `f32` (the loss value).
pub fn numerical_gradient(
    f: &dyn Fn(&Tensor) -> f32,
    x: &Tensor,
    eps: f32,
) -> Tensor {
    let mut grad_data = vec![0.0f32; x.data.len()];

    for i in 0..x.data.len() {
        let mut x_plus = x.data.clone();
        let mut x_minus = x.data.clone();
        x_plus[i] += eps;
        x_minus[i] -= eps;

        let loss_plus = f(&Tensor::new(x_plus, x.shape().to_vec()));
        let loss_minus = f(&Tensor::new(x_minus, x.shape().to_vec()));
        grad_data[i] = (loss_plus - loss_minus) / (2.0 * eps);
    }

    Tensor::new(grad_data, x.shape().to_vec())
}

/// Check that analytical and numerical gradients agree within tolerance.
///
/// Uses relative error: |a - n| / max(|a|, |n|, 1e-8) < tol
/// This handles near-zero gradients better than absolute difference.
pub fn check_gradient(analytical: &Tensor, numerical: &Tensor, tol: f32) -> bool {
    assert_eq!(
        analytical.shape(), numerical.shape(),
        "gradient shape mismatch: {:?} vs {:?}",
        analytical.shape(), numerical.shape()
    );

    for i in 0..analytical.data.len() {
        let a = analytical.data[i];
        let n = numerical.data[i];
        let denom = a.abs().max(n.abs()).max(1e-8);
        let rel_err = (a - n).abs() / denom;
        if rel_err > tol {
            return false;
        }
    }
    true
}

/// Convenience: compute analytical gradient via autograd, compare to numerical.
///
/// `build_loss` takes a `&Var` input and returns a scalar `Var` loss.
/// Returns `(passes, max_relative_error)`.
pub fn gradcheck(
    build_loss: &dyn Fn(&Var) -> Var,
    x_data: &Tensor,
    eps: f32,
    tol: f32,
) -> (bool, f32) {
    // Analytical
    let x = Var::new(x_data.clone(), true);
    let loss = build_loss(&x);
    loss.backward();
    let analytical = x.grad().expect("no gradient computed");

    // Numerical
    let numerical = numerical_gradient(
        &|t: &Tensor| {
            let v = Var::new(t.clone(), false);
            let l = build_loss(&v);
            l.tensor().data[0]
        },
        x_data,
        eps,
    );

    let mut max_rel_err = 0.0f32;
    for i in 0..analytical.data.len() {
        let a = analytical.data[i];
        let n = numerical.data[i];
        let denom = a.abs().max(n.abs()).max(1e-8);
        let rel_err = (a - n).abs() / denom;
        max_rel_err = max_rel_err.max(rel_err);
    }

    (max_rel_err < tol, max_rel_err)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_numerical_gradient_quadratic() {
        // f(x) = x^2, df/dx = 2x
        // For x = [3.0], f(x) = 9, df/dx = 6
        let x = Tensor::new(vec![3.0], vec![1]);
        let grad = numerical_gradient(
            &|t: &Tensor| t.data[0] * t.data[0],
            &x,
            1e-4,
        );
        assert!((grad.data[0] - 6.0).abs() < 1e-2);
    }

    #[test]
    fn test_check_gradient_pass() {
        let a = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]);
        let b = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]);
        assert!(check_gradient(&a, &b, 1e-5));
    }

    #[test]
    fn test_check_gradient_fail() {
        let a = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]);
        let b = Tensor::new(vec![1.0, 2.0, 4.0], vec![3]);
        assert!(!check_gradient(&a, &b, 1e-5));
    }

    // --- gradcheck on every Var operation ---

    #[test]
    fn gradcheck_add() {
        let x = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]);
        let (ok, err) = gradcheck(
            &|v| {
                let b = Var::new(Tensor::new(vec![4.0, 5.0, 6.0], vec![3]), false);
                v.add(&b).sum()
            },
            &x, 1e-3, 1e-2,
        );
        assert!(ok, "gradcheck_add failed: max_rel_err={}", err);
    }

    #[test]
    fn gradcheck_sub() {
        let x = Tensor::new(vec![5.0, 3.0, 1.0], vec![3]);
        let (ok, err) = gradcheck(
            &|v| {
                let b = Var::new(Tensor::new(vec![1.0, 2.0, 3.0], vec![3]), false);
                v.sub(&b).sum()
            },
            &x, 1e-3, 1e-2,
        );
        assert!(ok, "gradcheck_sub failed: max_rel_err={}", err);
    }

    #[test]
    fn gradcheck_mul() {
        let x = Tensor::new(vec![2.0, 3.0, 4.0], vec![3]);
        let (ok, err) = gradcheck(
            &|v| {
                let b = Var::new(Tensor::new(vec![5.0, 6.0, 7.0], vec![3]), false);
                v.mul(&b).sum()
            },
            &x, 1e-3, 1e-2,
        );
        assert!(ok, "gradcheck_mul failed: max_rel_err={}", err);
    }

    #[test]
    fn gradcheck_div() {
        let x = Tensor::new(vec![10.0, 20.0, 30.0], vec![3]);
        let (ok, err) = gradcheck(
            &|v| {
                let b = Var::new(Tensor::new(vec![2.0, 4.0, 5.0], vec![3]), false);
                v.div(&b).sum()
            },
            &x, 1e-3, 1e-2,
        );
        assert!(ok, "gradcheck_div failed: max_rel_err={}", err);
    }

    #[test]
    fn gradcheck_div_denominator() {
        // Check gradient w.r.t. denominator: d(a/b)/db = -a/b²
        let x = Tensor::new(vec![2.0, 4.0, 5.0], vec![3]);
        let (ok, err) = gradcheck(
            &|v| {
                let a = Var::new(Tensor::new(vec![10.0, 20.0, 30.0], vec![3]), false);
                a.div(v).sum()
            },
            &x, 1e-3, 1e-2,
        );
        assert!(ok, "gradcheck_div_denominator failed: max_rel_err={}", err);
    }

    #[test]
    fn gradcheck_neg() {
        let x = Tensor::new(vec![2.0, -3.0, 4.0], vec![3]);
        let (ok, err) = gradcheck(&|v| v.neg().sum(), &x, 1e-3, 1e-2);
        assert!(ok, "gradcheck_neg failed: max_rel_err={}", err);
    }

    #[test]
    fn gradcheck_exp() {
        let x = Tensor::new(vec![0.0, 0.5, 1.0], vec![3]);
        let (ok, err) = gradcheck(&|v| v.exp().sum(), &x, 1e-3, 1e-2);
        assert!(ok, "gradcheck_exp failed: max_rel_err={}", err);
    }

    #[test]
    fn gradcheck_log() {
        let x = Tensor::new(vec![1.0, 2.0, 4.0], vec![3]);
        let (ok, err) = gradcheck(&|v| v.log().sum(), &x, 1e-3, 1e-2);
        assert!(ok, "gradcheck_log failed: max_rel_err={}", err);
    }

    #[test]
    fn gradcheck_sqrt() {
        let x = Tensor::new(vec![1.0, 4.0, 9.0], vec![3]);
        let (ok, err) = gradcheck(&|v| v.sqrt().sum(), &x, 1e-3, 1e-2);
        assert!(ok, "gradcheck_sqrt failed: max_rel_err={}", err);
    }

    #[test]
    fn gradcheck_pow() {
        let x = Tensor::new(vec![2.0, 3.0, 4.0], vec![3]);
        let (ok, err) = gradcheck(&|v| v.pow(3.0).sum(), &x, 1e-3, 1e-2);
        assert!(ok, "gradcheck_pow failed: max_rel_err={}", err);
    }

    #[test]
    fn gradcheck_matmul() {
        let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let (ok, err) = gradcheck(
            &|v| {
                let b = Var::new(
                    Tensor::new(vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0], vec![3, 2]),
                    false,
                );
                v.matmul(&b).sum()
            },
            &x, 1e-3, 1e-2,
        );
        assert!(ok, "gradcheck_matmul failed: max_rel_err={}", err);
    }

    #[test]
    fn gradcheck_mul_scalar() {
        let x = Tensor::new(vec![2.0, 3.0, 4.0], vec![3]);
        let (ok, err) = gradcheck(&|v| v.mul_scalar(5.0).sum(), &x, 1e-3, 1e-2);
        assert!(ok, "gradcheck_mul_scalar failed: max_rel_err={}", err);
    }

    #[test]
    fn gradcheck_transpose() {
        let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let (ok, err) = gradcheck(&|v| v.transpose().sum(), &x, 1e-3, 1e-2);
        assert!(ok, "gradcheck_transpose failed: max_rel_err={}", err);
    }

    #[test]
    fn gradcheck_softmax() {
        let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let (ok, err) = gradcheck(
            &|v| {
                // softmax output sums to 1 per row, so sum(softmax) is just num_rows
                // Use a weighted sum to get a non-trivial gradient
                let s = v.softmax(1);
                let w = Var::new(
                    Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]),
                    false,
                );
                s.mul(&w).sum()
            },
            &x, 1e-3, 1e-2,
        );
        assert!(ok, "gradcheck_softmax failed: max_rel_err={}", err);
    }

    #[test]
    fn gradcheck_softmax_dim0() {
        let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let (ok, err) = gradcheck(
            &|v| {
                let s = v.softmax(0);
                let w = Var::new(
                    Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]),
                    false,
                );
                s.mul(&w).sum()
            },
            &x, 1e-3, 1e-2,
        );
        assert!(ok, "gradcheck_softmax_dim0 failed: max_rel_err={}", err);
    }

    // --- chain of operations ---

    #[test]
    fn gradcheck_chain_exp_mul_sum() {
        let x = Tensor::new(vec![0.5, 1.0, 1.5], vec![3]);
        let (ok, err) = gradcheck(
            &|v| {
                let e = v.exp();
                let w = Var::new(Tensor::new(vec![2.0, 3.0, 1.0], vec![3]), false);
                e.mul(&w).sum()
            },
            &x, 1e-3, 1e-2,
        );
        assert!(ok, "gradcheck_chain failed: max_rel_err={}", err);
    }

    #[test]
    fn gradcheck_chain_matmul_softmax() {
        let x = Tensor::new(vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6], vec![2, 3]);
        let (ok, err) = gradcheck(
            &|v| {
                let w = Var::new(
                    Tensor::new(vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6], vec![3, 2]),
                    false,
                );
                let h = v.matmul(&w); // [2, 2]
                let s = h.softmax(1);
                // Weight the softmax output so gradient is non-trivial
                // (sum(softmax) = const, so use a weighted sum)
                let weights = Var::new(
                    Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]),
                    false,
                );
                s.mul(&weights).sum()
            },
            &x, 1e-3, 5e-2,
        );
        assert!(ok, "gradcheck_chain_matmul_softmax failed: max_rel_err={}", err);
    }
}
