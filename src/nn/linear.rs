use crate::tensor::Tensor;
use crate::autograd::Var;

/// Fully-connected linear layer: y = x @ W^T + b
///
/// Weight shape: [out_features, in_features]
/// Bias shape: [out_features] (optional)
///
/// Input x can be any shape [..., in_features], output is [..., out_features].
/// Currently supports 2D input [batch, in_features] -> [batch, out_features].
pub struct Linear {
    pub weight: Var, // [out_features, in_features]
    pub bias: Option<Var>, // [out_features]
}

impl Linear {
    /// Create a new linear layer with Xavier normal initialization.
    pub fn new(in_features: usize, out_features: usize, bias: bool) -> Self {
        let weight = Var::new(
            Tensor::xaviern(&[out_features, in_features]),
            true,
        );
        let bias = if bias {
            Some(Var::new(Tensor::zeros(&[out_features]), true))
        } else {
            None
        };
        Linear { weight, bias }
    }

    /// Forward pass: y = x @ W^T + b
    ///
    /// Supports 2D input [batch, in_features] -> [batch, out_features]
    /// and 3D input [batch, seq, in_features] -> [batch, seq, out_features]
    /// (flattens leading dims, applies linear, then restores shape).
    pub fn forward(&self, x: &Var) -> Var {
        let shape = x.tensor().shape().to_vec();
        let ndim = shape.len();

        if ndim == 2 {
            let wt = self.weight.transpose(); // [in_features, out_features]
            let out = x.matmul(&wt);
            match &self.bias {
                Some(b) => out.add(b),
                None => out,
            }
        } else if ndim == 3 {
            let batch = shape[0];
            let seq = shape[1];
            let in_f = shape[2];

            // Flatten to 2D: [batch*seq, in_features]
            let flat = x.reshape(&[batch * seq, in_f]);
            let wt = self.weight.transpose();
            let out = flat.matmul(&wt); // [batch*seq, out_features]
            let out_f = out.tensor().shape()[1];

            // Restore to 3D: [batch, seq, out_features]
            let out3d = out.reshape(&[batch, seq, out_f]);
            match &self.bias {
                Some(b) => out3d.add(b),
                None => out3d,
            }
        } else {
            panic!("Linear::forward expects 2D or 3D input, got {}D", ndim);
        }
    }

    /// Collect all trainable parameters.
    pub fn parameters(&self) -> Vec<Var> {
        let mut params = vec![self.weight.clone()];
        if let Some(b) = &self.bias {
            params.push(b.clone());
        }
        params
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_forward_shape() {
        let layer = Linear::new(4, 3, true);
        let x = Var::new(Tensor::randn(&[2, 4]), false);
        let y = layer.forward(&x);
        assert_eq!(y.tensor().shape(), &[2, 3]);
    }

    #[test]
    fn test_forward_shape_no_bias() {
        let layer = Linear::new(5, 2, false);
        let x = Var::new(Tensor::randn(&[3, 5]), false);
        let y = layer.forward(&x);
        assert_eq!(y.tensor().shape(), &[3, 2]);
    }

    #[test]
    fn test_gradients_flow_to_weight_and_bias() {
        let layer = Linear::new(3, 2, true);
        let x = Var::new(Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]), true);
        let y = layer.forward(&x);
        let loss = y.sum();
        loss.backward();

        // Weight should have gradient
        let wg = layer.weight.grad().expect("weight has no gradient");
        assert_eq!(wg.shape(), &[2, 3]);

        // Bias should have gradient
        let bg = layer.bias.as_ref().unwrap().grad().expect("bias has no gradient");
        assert_eq!(bg.shape(), &[2]);

        // Input should have gradient
        let xg = x.grad().expect("input has no gradient");
        assert_eq!(xg.shape(), &[2, 3]);
    }

    #[test]
    fn test_gradients_flow_no_bias() {
        let layer = Linear::new(3, 2, false);
        let x = Var::new(Tensor::randn(&[2, 3]), true);
        let loss = layer.forward(&x).sum();
        loss.backward();

        assert!(layer.weight.grad().is_some());
        assert!(layer.bias.is_none());
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_known_values() {
        // Manual computation: x @ W^T + b
        // W = [[1, 0, 0], [0, 1, 0]]  (2x3)
        // b = [10, 20]
        // x = [[1, 2, 3]]  (1x3)
        // y = [[1, 2]] + [[10, 20]] = [[11, 22]]
        let layer = Linear {
            weight: Var::new(
                Tensor::new(vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0], vec![2, 3]),
                true,
            ),
            bias: Some(Var::new(Tensor::new(vec![10.0, 20.0], vec![2]), true)),
        };
        let x = Var::new(Tensor::new(vec![1.0, 2.0, 3.0], vec![1, 3]), false);
        let y = layer.forward(&x);
        let yd = y.tensor();
        assert_eq!(yd.shape(), &[1, 2]);
        assert!((yd.data[0] - 11.0).abs() < 1e-5);
        assert!((yd.data[1] - 22.0).abs() < 1e-5);
    }

    #[test]
    fn test_parameters_count() {
        let with_bias = Linear::new(4, 3, true);
        assert_eq!(with_bias.parameters().len(), 2);

        let without_bias = Linear::new(4, 3, false);
        assert_eq!(without_bias.parameters().len(), 1);
    }

    #[test]
    fn test_gradcheck_linear() {
        use crate::autograd::gradcheck;

        // Check gradient w.r.t. input
        let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        // Use fixed weights so the function is deterministic
        let w_data = Tensor::new(vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6], vec![2, 3]);
        let b_data = Tensor::new(vec![0.01, 0.02], vec![2]);

        let (ok, err) = gradcheck(
            &|v| {
                let w = Var::new(w_data.clone(), false);
                let b = Var::new(b_data.clone(), false);
                let wt = w.transpose();
                let out = v.matmul(&wt);
                out.add(&b).sum()
            },
            &x, 5e-4, 1e-2,
        );
        assert!(ok, "linear gradcheck (input) failed: max_rel_err={}", err);
    }

    #[test]
    fn test_gradcheck_linear_weight() {
        use crate::autograd::gradcheck;

        // Check gradient w.r.t. weight
        let w = Tensor::new(vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6], vec![2, 3]);
        let x_data = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);

        let (ok, err) = gradcheck(
            &|v| {
                let x = Var::new(x_data.clone(), false);
                let wt = v.transpose();
                x.matmul(&wt).sum()
            },
            &w, 5e-4, 1e-2,
        );
        assert!(ok, "linear gradcheck (weight) failed: max_rel_err={}", err);
    }
}
