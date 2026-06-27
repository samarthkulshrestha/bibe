use crate::autograd::Var;

/// Global L2 norm of all parameter gradients, treated as one flat vector.
pub fn grad_global_norm(params: &[Var]) -> f32 {
    let mut sum_sq = 0.0f32;
    for p in params {
        if let Some(g) = p.grad() {
            for &x in &g.data {
                sum_sq += x * x;
            }
        }
    }
    sum_sq.sqrt()
}

/// Clip parameter gradients by their global norm.
///
/// If the global gradient norm exceeds `max_norm`, every gradient is scaled by
/// `max_norm / norm` so the clipped global norm equals `max_norm`; otherwise
/// gradients are left unchanged. Returns the pre-clip global norm (useful for
/// monitoring training stability).
pub fn clip_grad_norm(params: &[Var], max_norm: f32) -> f32 {
    let norm = grad_global_norm(params);
    if norm > max_norm {
        let scale = max_norm / (norm + 1e-6);
        for p in params {
            p.scale_grad(scale);
        }
    }
    norm
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::Tensor;

    /// A parameter whose gradient becomes `grad` after backward.
    fn param_with_grad(grad: &[f32]) -> Var {
        let p = Var::new(Tensor::new(vec![0.0; grad.len()], vec![grad.len()]), true);
        let g = Var::new(Tensor::new(grad.to_vec(), vec![grad.len()]), false);
        // loss = sum(p * g) -> d loss / d p = g
        let loss = p.mul(&g).sum();
        loss.backward();
        p
    }

    #[test]
    fn test_global_norm() {
        // grads [3,4] and [12] -> sqrt(9+16+144) = 13
        let params = vec![param_with_grad(&[3.0, 4.0]), param_with_grad(&[12.0])];
        assert!((grad_global_norm(&params) - 13.0).abs() < 1e-4);
    }

    #[test]
    fn test_clip_scales_down_when_exceeding() {
        let params = vec![param_with_grad(&[3.0, 4.0])]; // norm 5
        let pre = clip_grad_norm(&params, 1.0);
        assert!((pre - 5.0).abs() < 1e-4, "should return pre-clip norm");
        // After clipping the global norm should be ~max_norm.
        assert!((grad_global_norm(&params) - 1.0).abs() < 1e-3);
    }

    #[test]
    fn test_no_clip_when_within_threshold() {
        let params = vec![param_with_grad(&[3.0, 4.0])]; // norm 5
        clip_grad_norm(&params, 10.0);
        // Unchanged.
        assert!((grad_global_norm(&params) - 5.0).abs() < 1e-4);
        assert_eq!(params[0].grad().unwrap().data, vec![3.0, 4.0]);
    }
}
