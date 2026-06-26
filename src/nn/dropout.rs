use rand::Rng;

use crate::autograd::Var;
use crate::tensor::Tensor;

/// Inverted dropout.
///
/// During training each element is zeroed independently with probability `p`
/// and surviving elements are scaled by `1 / (1 - p)`, so the expected value
/// of each activation is preserved and no scaling is needed at evaluation
/// time. During evaluation the input passes through unchanged.
pub struct Dropout {
    p: f32,
}

impl Dropout {
    /// Create a dropout layer that zeroes elements with probability `p`.
    pub fn new(p: f32) -> Self {
        assert!((0.0..1.0).contains(&p), "dropout probability must be in [0, 1)");
        Dropout { p }
    }

    /// Apply dropout. When `training` is false (or `p == 0`) this is the
    /// identity; otherwise it applies an inverted-dropout mask.
    pub fn forward(&self, x: &Var, training: bool) -> Var {
        if !training || self.p == 0.0 {
            return x.clone();
        }

        let scale = 1.0 / (1.0 - self.p);
        let shape = x.tensor().shape().to_vec();
        let size: usize = shape.iter().product();

        let mut rng = rand::rng();
        let mask_data: Vec<f32> = (0..size)
            .map(|_| if rng.random::<f32>() < self.p { 0.0 } else { scale })
            .collect();

        // The mask is a constant (no gradient); multiplying by it routes the
        // gradient straight through for kept positions and zeroes it for
        // dropped ones.
        let mask = Var::new(Tensor::new(mask_data, shape), false);
        x.mul(&mask)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_eval_mode_is_identity() {
        let drop = Dropout::new(0.5);
        let x = Var::new(Tensor::randn(&[4, 8]), false);
        let y = drop.forward(&x, false);
        assert_eq!(x.tensor().data, y.tensor().data);
    }

    #[test]
    fn test_zero_probability_is_identity() {
        let drop = Dropout::new(0.0);
        let x = Var::new(Tensor::randn(&[4, 8]), false);
        let y = drop.forward(&x, true);
        assert_eq!(x.tensor().data, y.tensor().data);
    }

    #[test]
    fn test_training_drops_approximately_p_fraction() {
        let p = 0.5;
        let drop = Dropout::new(p);
        let x = Var::new(Tensor::ones(&[100, 100]), false);
        let y = drop.forward(&x, true);
        let zeros = y.tensor().data.iter().filter(|&&v| v == 0.0).count();
        let frac = zeros as f32 / 10_000.0;
        assert!((frac - p).abs() < 0.05, "dropped fraction {frac} far from {p}");
    }

    #[test]
    fn test_surviving_elements_are_scaled() {
        let p = 0.5;
        let drop = Dropout::new(p);
        let x = Var::new(Tensor::ones(&[50, 50]), false);
        let y = drop.forward(&x, true);
        // Each element is either 0 or scaled to 1/(1-p) = 2.0.
        let scale = 1.0 / (1.0 - p);
        for &v in &y.tensor().data {
            assert!(
                v == 0.0 || (v - scale).abs() < 1e-5,
                "value {v} is neither 0 nor {scale}"
            );
        }
    }

    #[test]
    fn test_expected_value_preserved() {
        let p = 0.3;
        let drop = Dropout::new(p);
        let x = Var::new(Tensor::ones(&[200, 200]), false);
        let y = drop.forward(&x, true);
        // Mean over many elements should stay close to the input mean (1.0).
        let mean: f32 = y.tensor().data.iter().sum::<f32>() / 40_000.0;
        assert!((mean - 1.0).abs() < 0.05, "mean {mean} drifted from 1.0");
    }

    #[test]
    fn test_gradient_matches_mask() {
        let drop = Dropout::new(0.5);
        let x = Var::new(Tensor::ones(&[20, 20]), true);
        let y = drop.forward(&x, true);
        let loss = y.sum();
        loss.backward();

        let out = y.tensor();
        let grad = x.grad().unwrap();
        let scale = 2.0_f32;
        for i in 0..out.data.len() {
            if out.data[i] == 0.0 {
                assert_eq!(grad.data[i], 0.0, "dropped position should have zero grad");
            } else {
                assert!((grad.data[i] - scale).abs() < 1e-5, "kept grad should be {scale}");
            }
        }
    }
}
