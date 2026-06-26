use crate::autograd::Var;
use crate::nn::Linear;

/// Position-wise feed-forward network applied independently at each position.
///
/// ```text
/// FFN(x) = W2 · GeLU(W1 · x + b1) + b2
/// ```
///
/// The hidden dimension `d_ff` is typically 4× `d_model`.
pub struct PositionwiseFFN {
    pub linear1: Linear,
    pub linear2: Linear,
}

impl PositionwiseFFN {
    /// Create an FFN mapping `d_model -> d_ff -> d_model`.
    pub fn new(d_model: usize, d_ff: usize) -> Self {
        PositionwiseFFN {
            linear1: Linear::new(d_model, d_ff, true),
            linear2: Linear::new(d_ff, d_model, true),
        }
    }

    /// Forward pass: expand to `d_ff`, apply GeLU, project back to `d_model`.
    pub fn forward(&self, x: &Var) -> Var {
        let hidden = self.linear1.forward(x).gelu();
        self.linear2.forward(&hidden)
    }

    /// Collect all trainable parameters.
    pub fn parameters(&self) -> Vec<Var> {
        let mut params = self.linear1.parameters();
        params.extend(self.linear2.parameters());
        params
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::Tensor;

    #[test]
    fn test_preserves_shape_3d() {
        let ffn = PositionwiseFFN::new(8, 32);
        let x = Var::new(Tensor::randn(&[2, 5, 8]), false);
        let y = ffn.forward(&x);
        assert_eq!(y.tensor().shape(), &[2, 5, 8]);
    }

    #[test]
    fn test_parameter_count() {
        // Two linear layers, each with weight + bias.
        let ffn = PositionwiseFFN::new(8, 32);
        assert_eq!(ffn.parameters().len(), 4);
    }

    #[test]
    fn test_transforms_input() {
        // A real FFN is not the identity; its output should differ from input.
        let ffn = PositionwiseFFN::new(8, 32);
        let x = Var::new(Tensor::randn(&[1, 3, 8]), false);
        let y = ffn.forward(&x);
        let xt = x.tensor();
        let yt = y.tensor();
        let differs = xt.data.iter().zip(yt.data.iter()).any(|(a, b)| (a - b).abs() > 1e-4);
        assert!(differs, "FFN output should differ from its input");
    }

    #[test]
    fn test_all_parameters_receive_gradients() {
        let ffn = PositionwiseFFN::new(8, 16);
        let x = Var::new(Tensor::randn(&[2, 4, 8]), true);
        let loss = ffn.forward(&x).sum();
        loss.backward();

        assert!(x.grad().is_some(), "input has no gradient");
        for (i, p) in ffn.parameters().iter().enumerate() {
            assert!(p.grad().is_some(), "parameter {i} has no gradient");
        }
    }
}
