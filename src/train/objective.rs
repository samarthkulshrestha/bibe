use crate::autograd::Var;

use super::loss::contrastive_loss;

/// Clamp bound keeping predicted probabilities clear of the log singularity.
const EPS: f32 = 1e-7;

/// Mean focal loss over the real (non-padded) positions of a batch.
///
/// `pred`, `target`, and `mask` are all `[batch, seq]`; `mask` is 1.0 for real
/// events and 0.0 for padding. Padded positions contribute neither to the
/// numerator nor the denominator.
pub fn masked_focal_loss(
    pred: &Var,
    target: &Var,
    mask: &Var,
    alpha: f32,
    gamma: f32,
) -> Var {
    let p = pred.clamp(EPS, 1.0 - EPS);
    let one_minus_p = p.neg().add_scalar(1.0);
    let one_minus_y = target.neg().add_scalar(1.0);

    // Per-position focal loss (>= 0), not yet reduced.
    let pos = one_minus_p
        .pow(gamma)
        .mul(target)
        .mul(&p.log())
        .mul_scalar(alpha);
    let neg = p
        .pow(gamma)
        .mul(&one_minus_y)
        .mul(&one_minus_p.log())
        .mul_scalar(1.0 - alpha);
    let per_pos = pos.add(&neg).mul_scalar(-1.0);

    // Average over real positions only.
    let real_count = mask.tensor().data.iter().sum::<f32>().max(1.0);
    per_pos.mul(mask).sum().mul_scalar(1.0 / real_count)
}

/// Mean-pool encoder hidden states over real positions into one vector per
/// trace.
///
/// `hidden` is `[batch, seq, d_model]`, `mask` is `[batch, seq]`. Returns
/// `[batch, d_model]`: the mask-weighted average over the sequence, so padding
/// does not dilute the trace representation.
pub fn masked_mean_pool(hidden: &Var, mask: &Var) -> Var {
    let shape = hidden.tensor().shape().to_vec();
    let (b, s, d) = (shape[0], shape[1], shape[2]);

    let mask_exp = mask.reshape(&[b, s, 1]);
    // Σ_s hidden·mask divided by Σ_s mask, both via mean (the /S factors cancel).
    let summed = hidden.mul(&mask_exp).mean(1); // [b, 1, d]
    let count = mask_exp.mean(1); // [b, 1, 1]
    let pooled = summed.div(&count.add_scalar(EPS)); // [b, 1, d]
    pooled.reshape(&[b, d])
}

/// Batch-level contrastive trace loss.
///
/// Treats each anomalous trace's pooled representation as an anchor, another
/// anomalous trace as its positive, and all normal traces as negatives, then
/// averages the InfoNCE loss over anchors. Returns `None` when the batch lacks
/// the needed composition (fewer than two anomalous traces or no normal
/// trace), so the caller can skip the term for that batch.
pub fn batch_contrastive_loss(
    pooled: &Var,
    is_anomalous: &[bool],
    temperature: f32,
) -> Option<Var> {
    let anomalous: Vec<usize> = (0..is_anomalous.len()).filter(|&i| is_anomalous[i]).collect();
    let normal: Vec<usize> = (0..is_anomalous.len()).filter(|&i| !is_anomalous[i]).collect();

    // Need at least two anomalous traces (anchor + positive) and one negative.
    if anomalous.len() < 2 || normal.is_empty() {
        return None;
    }

    let rows: Vec<Var> = (0..is_anomalous.len()).map(|i| pooled.select_row(i)).collect();
    let negatives: Vec<Var> = normal.iter().map(|&j| rows[j].clone()).collect();

    let mut total: Option<Var> = None;
    for (k, &a) in anomalous.iter().enumerate() {
        let positive = anomalous[(k + 1) % anomalous.len()];
        let term = contrastive_loss(&rows[a], &rows[positive], &negatives, temperature);
        total = Some(match total {
            Some(t) => t.add(&term),
            None => term,
        });
    }

    total.map(|t| t.mul_scalar(1.0 / anomalous.len() as f32))
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::loss::focal_loss;
    use crate::tensor::Tensor;

    #[test]
    fn test_masked_focal_equals_focal_when_all_real() {
        let pred = Var::new(Tensor::new(vec![0.6, 0.3, 0.8, 0.4], vec![2, 2]), false);
        let target = Var::new(Tensor::new(vec![1.0, 0.0, 1.0, 0.0], vec![2, 2]), false);
        let mask = Var::new(Tensor::new(vec![1.0, 1.0, 1.0, 1.0], vec![2, 2]), false);
        let masked = masked_focal_loss(&pred, &target, &mask, 0.75, 2.0).tensor().data[0];
        let plain = focal_loss(&pred, &target, 0.75, 2.0).tensor().data[0];
        assert!((masked - plain).abs() < 1e-6, "{masked} vs {plain}");
    }

    #[test]
    fn test_masked_focal_ignores_padding() {
        // Two real positions plus a padded one with a deliberately bad pred.
        let real = masked_focal_loss(
            &Var::new(Tensor::new(vec![0.6, 0.3], vec![1, 2]), false),
            &Var::new(Tensor::new(vec![1.0, 0.0], vec![1, 2]), false),
            &Var::new(Tensor::new(vec![1.0, 1.0], vec![1, 2]), false),
            0.75,
            2.0,
        )
        .tensor()
        .data[0];
        let with_pad = masked_focal_loss(
            &Var::new(Tensor::new(vec![0.6, 0.3, 0.001], vec![1, 3]), false),
            &Var::new(Tensor::new(vec![1.0, 0.0, 1.0], vec![1, 3]), false),
            &Var::new(Tensor::new(vec![1.0, 1.0, 0.0], vec![1, 3]), false),
            0.75,
            2.0,
        )
        .tensor()
        .data[0];
        assert!((real - with_pad).abs() < 1e-6, "padding changed the loss: {real} vs {with_pad}");
    }

    #[test]
    fn test_masked_focal_gradient_flows() {
        let pred = Var::new(Tensor::new(vec![0.6, 0.3], vec![1, 2]), true);
        let target = Var::new(Tensor::new(vec![1.0, 0.0], vec![1, 2]), false);
        let mask = Var::new(Tensor::new(vec![1.0, 1.0], vec![1, 2]), false);
        masked_focal_loss(&pred, &target, &mask, 0.75, 2.0).backward();
        assert!(pred.grad().is_some());
    }

    #[test]
    fn test_mean_pool_shape_and_value() {
        // hidden [1,2,2] rows [1,1] and [3,3].
        let hidden = Var::new(Tensor::new(vec![1.0, 1.0, 3.0, 3.0], vec![1, 2, 2]), false);
        // Full mask -> average of the two rows = [2,2].
        let full = masked_mean_pool(&hidden, &Var::new(Tensor::new(vec![1.0, 1.0], vec![1, 2]), false));
        assert_eq!(full.tensor().shape(), &[1, 2]);
        for v in &full.tensor().data {
            assert!((v - 2.0).abs() < 1e-5, "{v}");
        }
        // Mask out second position -> just the first row [1,1].
        let first = masked_mean_pool(&hidden, &Var::new(Tensor::new(vec![1.0, 0.0], vec![1, 2]), false));
        for v in &first.tensor().data {
            assert!((v - 1.0).abs() < 1e-5, "{v}");
        }
    }

    #[test]
    fn test_mean_pool_gradient_flows() {
        let hidden = Var::new(Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]), true);
        let mask = Var::new(Tensor::new(vec![1.0, 1.0], vec![1, 2]), false);
        masked_mean_pool(&hidden, &mask).sum().backward();
        assert!(hidden.grad().is_some());
    }

    #[test]
    fn test_contrastive_none_without_composition() {
        let pooled = Var::new(Tensor::new(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2]), false);
        // Only one anomalous -> None.
        assert!(batch_contrastive_loss(&pooled, &[true, false], 0.5).is_none());
        // No normal -> None.
        assert!(batch_contrastive_loss(&pooled, &[true, true], 0.5).is_none());
    }

    #[test]
    fn test_contrastive_some_and_differentiable() {
        // 2 anomalous + 1 normal.
        let pooled = Var::new(
            Tensor::new(vec![1.0, 0.0, 0.9, 0.1, 0.0, 1.0], vec![3, 2]),
            true,
        );
        let loss = batch_contrastive_loss(&pooled, &[true, true, false], 0.5);
        assert!(loss.is_some(), "valid composition should yield a loss");
        let loss = loss.unwrap();
        assert!(loss.tensor().data[0] > 0.0);
        loss.backward();
        assert!(pooled.grad().is_some());
    }
}
