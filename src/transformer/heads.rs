use crate::autograd::Var;
use crate::nn::Linear;

/// Per-position anomaly detection head.
///
/// Maps each position's hidden vector to a single logit and applies a sigmoid,
/// producing `P(anomaly)` for every event in the trace.
pub struct AnomalyHead {
    classifier: Linear,
}

impl AnomalyHead {
    /// Create a head projecting `d_model` features to a single score.
    pub fn new(d_model: usize) -> Self {
        AnomalyHead {
            classifier: Linear::new(d_model, 1, true),
        }
    }

    /// Forward pass: `[batch, seq, d_model] -> [batch, seq]` anomaly
    /// probabilities in `(0, 1)`.
    pub fn forward(&self, hidden: &Var) -> Var {
        let shape = hidden.tensor().shape().to_vec();
        let (b, s) = (shape[0], shape[1]);
        // [batch, seq, d_model] -> [batch, seq, 1] -> sigmoid -> [batch, seq]
        let logits = self.classifier.forward(hidden);
        logits.sigmoid().reshape(&[b, s])
    }

    /// Collect all trainable parameters.
    pub fn parameters(&self) -> Vec<Var> {
        self.classifier.parameters()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::Tensor;

    #[test]
    fn test_output_shape() {
        let head = AnomalyHead::new(16);
        let hidden = Var::new(Tensor::randn(&[2, 5, 16]), false);
        let scores = head.forward(&hidden);
        assert_eq!(scores.tensor().shape(), &[2, 5]);
    }

    #[test]
    fn test_scores_are_probabilities() {
        let head = AnomalyHead::new(16);
        let hidden = Var::new(Tensor::randn(&[3, 4, 16]), false);
        let scores = head.forward(&hidden);
        for &v in &scores.tensor().data {
            assert!(v > 0.0 && v < 1.0, "score {v} is not a probability");
        }
    }

    #[test]
    fn test_gradients_flow() {
        let head = AnomalyHead::new(8);
        let hidden = Var::new(Tensor::randn(&[2, 3, 8]), true);
        let loss = head.forward(&hidden).sum();
        loss.backward();

        assert!(hidden.grad().is_some(), "input has no gradient");
        for (i, p) in head.parameters().iter().enumerate() {
            assert!(p.grad().is_some(), "parameter {i} has no gradient");
        }
    }
}
