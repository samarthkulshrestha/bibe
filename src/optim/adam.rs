use crate::autograd::Var;
use crate::tensor::Tensor;

/// Adam optimizer (Kingma & Ba, 2014).
///
/// Maintains per-parameter first and second moment estimates and applies
/// bias-corrected updates:
///
/// ```text
/// m_t = β1 * m_{t-1} + (1 - β1) * g
/// v_t = β2 * v_{t-1} + (1 - β2) * g²
/// m̂  = m_t / (1 - β1^t)
/// v̂  = v_t / (1 - β2^t)
/// θ   = θ - lr * m̂ / (√v̂ + ε)
/// ```
pub struct Adam {
    params: Vec<Var>,
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    m: Vec<Tensor>,
    v: Vec<Tensor>,
    t: u64,
}

impl Adam {
    /// Construct with the conventional defaults (β1=0.9, β2=0.999, ε=1e-8).
    pub fn new(params: Vec<Var>, lr: f32) -> Self {
        Self::with_hyperparams(params, lr, 0.9, 0.999, 1e-8)
    }

    /// Construct with explicit hyperparameters.
    pub fn with_hyperparams(
        params: Vec<Var>,
        lr: f32,
        beta1: f32,
        beta2: f32,
        eps: f32,
    ) -> Self {
        let m = params
            .iter()
            .map(|p| Tensor::zeros(p.tensor().shape()))
            .collect();
        let v = params
            .iter()
            .map(|p| Tensor::zeros(p.tensor().shape()))
            .collect();
        Adam {
            params,
            lr,
            beta1,
            beta2,
            eps,
            m,
            v,
            t: 0,
        }
    }

    /// Set the learning rate (used to apply a schedule between steps).
    pub fn set_lr(&mut self, lr: f32) {
        self.lr = lr;
    }

    /// The current learning rate.
    pub fn lr(&self) -> f32 {
        self.lr
    }

    /// Clear gradients on all tracked parameters.
    pub fn zero_grad(&self) {
        for p in &self.params {
            p.zero_grad();
        }
    }

    /// Apply one optimization step using the gradients accumulated on each
    /// parameter. Parameters without a gradient are skipped.
    pub fn step(&mut self) {
        self.t += 1;
        let bias_c1 = 1.0 - self.beta1.powi(self.t as i32);
        let bias_c2 = 1.0 - self.beta2.powi(self.t as i32);

        for (i, p) in self.params.iter().enumerate() {
            let grad = match p.grad() {
                Some(g) => g,
                None => continue,
            };

            let m = &mut self.m[i];
            let v = &mut self.v[i];
            let (lr, beta1, beta2, eps) = (self.lr, self.beta1, self.beta2, self.eps);

            p.with_data_mut(|theta| {
                for j in 0..theta.data.len() {
                    let g = grad.data[j];
                    m.data[j] = beta1 * m.data[j] + (1.0 - beta1) * g;
                    v.data[j] = beta2 * v.data[j] + (1.0 - beta2) * g * g;
                    let m_hat = m.data[j] / bias_c1;
                    let v_hat = v.data[j] / bias_c2;
                    theta.data[j] -= lr * m_hat / (v_hat.sqrt() + eps);
                }
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: &[f32], b: &[f32], tol: f32) {
        assert_eq!(a.len(), b.len(), "length mismatch: {} vs {}", a.len(), b.len());
        for (i, (&x, &y)) in a.iter().zip(b.iter()).enumerate() {
            assert!(
                (x - y).abs() < tol,
                "element {} differs: {} vs {} (tol={})",
                i, x, y, tol
            );
        }
    }

    #[test]
    fn test_converges_on_quadratic() {
        // Minimize (w - target)^2; optimum at w == target.
        let w = Var::new(Tensor::new(vec![0.0, 0.0], vec![2]), true);
        let target = Var::new(Tensor::new(vec![3.0, -1.0], vec![2]), false);
        let mut opt = Adam::new(vec![w.clone()], 0.1);

        for _ in 0..500 {
            opt.zero_grad();
            let diff = w.sub(&target);
            let loss = diff.mul(&diff).sum();
            loss.backward();
            opt.step();
        }

        approx_eq(&w.tensor().data, &[3.0, -1.0], 1e-2);
    }

    #[test]
    fn test_first_step_uses_bias_correction() {
        // With bias correction the first update magnitude is ~lr regardless
        // of gradient scale, since m_hat / sqrt(v_hat) ≈ sign(g).
        let w = Var::new(Tensor::new(vec![0.0], vec![1]), true);
        let mut opt = Adam::new(vec![w.clone()], 0.1);

        // d(loss)/dw = 1000 (a deliberately large gradient).
        let loss = w.mul_scalar(1000.0).sum();
        loss.backward();
        opt.step();

        // Moves by -lr * sign(g) = -0.1, not a tiny step.
        approx_eq(&w.tensor().data, &[-0.1], 1e-4);
    }

    #[test]
    fn test_set_and_get_lr() {
        let p = Var::new(Tensor::new(vec![0.0], vec![1]), true);
        let mut opt = Adam::new(vec![p], 0.1);
        assert!((opt.lr() - 0.1).abs() < 1e-9);
        opt.set_lr(0.05);
        assert!((opt.lr() - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_skips_params_without_grad() {
        // A parameter that never participates in the loss has no gradient
        // and must be left untouched.
        let used = Var::new(Tensor::new(vec![0.0], vec![1]), true);
        let unused = Var::new(Tensor::new(vec![7.0], vec![1]), true);
        let target = Var::new(Tensor::new(vec![1.0], vec![1]), false);
        let mut opt = Adam::new(vec![used.clone(), unused.clone()], 0.1);

        opt.zero_grad();
        let diff = used.sub(&target);
        let loss = diff.mul(&diff).sum();
        loss.backward();
        opt.step();

        approx_eq(&unused.tensor().data, &[7.0], 1e-9);
        assert!(used.tensor().data[0] != 0.0, "used param should have moved");
    }
}
