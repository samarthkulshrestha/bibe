use crate::autograd::Var;

/// Clamp bound keeping predicted probabilities clear of the log singularity.
const EPS: f32 = 1e-7;

/// Mean binary cross-entropy between predicted probabilities and 0/1 targets.
///
/// ```text
/// BCE = -mean[ y·log(p) + (1-y)·log(1-p) ]
/// ```
///
/// `pred` holds probabilities in `(0, 1)`; `target` holds 0/1 labels and is
/// treated as a constant.
pub fn bce_loss(pred: &Var, target: &Var) -> Var {
    let n = pred.tensor().data.len() as f32;
    let p = pred.clamp(EPS, 1.0 - EPS);

    // y · log(p)
    let term_pos = target.mul(&p.log());
    // (1 - y) · log(1 - p)
    let one_minus_y = target.neg().add_scalar(1.0);
    let one_minus_p = p.neg().add_scalar(1.0);
    let term_neg = one_minus_y.mul(&one_minus_p.log());

    term_pos.add(&term_neg).sum().mul_scalar(-1.0 / n)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::autograd::gradcheck::gradcheck;
    use crate::tensor::Tensor;

    #[test]
    fn test_bce_uniform_prediction() {
        // p = 0.5 everywhere -> loss = -log(0.5) = ln 2 regardless of labels.
        let pred = Var::new(Tensor::new(vec![0.5, 0.5], vec![2]), false);
        let target = Var::new(Tensor::new(vec![1.0, 0.0], vec![2]), false);
        let loss = bce_loss(&pred, &target);
        assert!((loss.tensor().data[0] - 2.0_f32.ln()).abs() < 1e-5);
    }

    #[test]
    fn test_bce_confident_correct_is_small() {
        let pred = Var::new(Tensor::new(vec![0.999, 0.001], vec![2]), false);
        let target = Var::new(Tensor::new(vec![1.0, 0.0], vec![2]), false);
        let loss = bce_loss(&pred, &target);
        assert!(loss.tensor().data[0] < 1e-2, "confident-correct loss too large");
    }

    #[test]
    fn test_bce_confident_wrong_is_large() {
        let pred = Var::new(Tensor::new(vec![0.001, 0.999], vec![2]), false);
        let target = Var::new(Tensor::new(vec![1.0, 0.0], vec![2]), false);
        let loss = bce_loss(&pred, &target);
        assert!(loss.tensor().data[0] > 5.0, "confident-wrong loss too small");
    }

    #[test]
    fn test_bce_gradient_numeric() {
        let pred = Tensor::new(vec![0.6, 0.3, 0.8], vec![3]);
        let target = Var::new(Tensor::new(vec![1.0, 0.0, 1.0], vec![3]), false);
        let (ok, err) = gradcheck(&|p: &Var| bce_loss(p, &target), &pred, 1e-3, 2e-3);
        assert!(ok, "bce gradcheck failed, max rel err = {err}");
    }
}
