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

/// Attribution supervision: push each symptom query's attention toward the
/// labeled causal event.
///
/// `attention_weights` are the per-layer self-attention tensors
/// `[batch*num_heads, seq, seq]`. `supervised` lists `(window, symptom, cause)`
/// triples — typically only windows where the cause differs from the symptom.
/// For each layer and triple, the attention from the symptom query to the
/// cause is averaged over heads and penalized by `-log`, so minimizing the
/// loss drives that attention mass toward 1. Returns `None` when there is
/// nothing to supervise.
pub fn attribution_supervision_loss(
    attention_weights: &[Var],
    supervised: &[(usize, usize, usize)],
    num_heads: usize,
) -> Option<Var> {
    if supervised.is_empty() || attention_weights.is_empty() {
        return None;
    }

    let mut total: Option<Var> = None;
    let mut terms = 0usize;

    for attn in attention_weights {
        let shape = attn.tensor().shape().to_vec();
        let (bh, seq) = (shape[0], shape[1]);
        // Flatten [bh, seq, seq] -> [bh*seq, seq] so each (group, query) row is
        // addressable by select_row.
        let flat = attn.reshape(&[bh * seq, seq]);

        for &(window, symptom, cause) in supervised {
            // Average the symptom query's attention-to-cause over heads.
            let mut acc: Option<Var> = None;
            for h in 0..num_heads {
                let group = window * num_heads + h;
                let row = flat.select_row(group * seq + symptom); // [seq]
                let to_cause = row.reshape(&[seq, 1]).select_row(cause); // [1]
                acc = Some(match acc {
                    Some(a) => a.add(&to_cause),
                    None => to_cause,
                });
            }
            let avg = acc.unwrap().mul_scalar(1.0 / num_heads as f32);
            // -log(attention to cause): minimized as that attention -> 1.
            let term = avg.add_scalar(EPS).log().mul_scalar(-1.0);
            total = Some(match total {
                Some(t) => t.add(&term),
                None => term,
            });
            terms += 1;
        }
    }

    total.map(|t| t.mul_scalar(1.0 / terms as f32))
}

/// Attribution supervision applied directly to the differentiable attention
/// rollout `[batch, seq, seq]`.
///
/// For each `(window, symptom, cause)` triple, the rollout influence from the
/// symptom query to the cause is penalized by `-log`, so minimizing drives the
/// rollout — the value actually used for attribution at inference — to put mass
/// on the cause. Supervising the rollout rather than raw attention removes the
/// train/eval mismatch. Returns `None` when there is nothing to supervise.
pub fn rollout_supervision_loss(rollout: &Var, supervised: &[(usize, usize, usize)]) -> Option<Var> {
    if supervised.is_empty() {
        return None;
    }

    let shape = rollout.tensor().shape().to_vec();
    let seq = shape[1];
    let flat = rollout.reshape(&[shape[0] * seq, seq]);

    let mut total: Option<Var> = None;
    for &(window, symptom, cause) in supervised {
        let row = flat.select_row(window * seq + symptom); // [seq]
        let to_cause = row.reshape(&[seq, 1]).select_row(cause); // [1]
        let term = to_cause.add_scalar(EPS).log().mul_scalar(-1.0);
        total = Some(match total {
            Some(t) => t.add(&term),
            None => term,
        });
    }

    total.map(|t| t.mul_scalar(1.0 / supervised.len() as f32))
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::loss::focal_loss;
    use crate::tensor::Tensor;

    /// A [1, seq, seq] rollout whose query-row `q` is `row`, others uniform.
    fn rollout_with_query_row(seq: usize, q: usize, row: &[f32]) -> Tensor {
        let uniform = 1.0 / seq as f32;
        let mut data = vec![uniform; seq * seq];
        for (j, &v) in row.iter().enumerate() {
            data[q * seq + j] = v;
        }
        Tensor::new(data, vec![1, seq, seq])
    }

    #[test]
    fn test_rollout_supervision_none_when_empty() {
        let r = Var::new(Tensor::zeros(&[1, 3, 3]), false);
        assert!(rollout_supervision_loss(&r, &[]).is_none());
    }

    #[test]
    fn test_rollout_supervision_lower_when_pointing_at_cause() {
        let pointed = Var::new(rollout_with_query_row(3, 2, &[1.0, 0.0, 0.0]), false);
        let uniform = Var::new(rollout_with_query_row(3, 2, &[1.0 / 3.0; 3]), false);
        let lp = rollout_supervision_loss(&pointed, &[(0, 2, 0)]).unwrap().tensor().data[0];
        let lu = rollout_supervision_loss(&uniform, &[(0, 2, 0)]).unwrap().tensor().data[0];
        assert!(lp < lu, "pointing at cause should be lower: {lp} vs {lu}");
        assert!(lp < 1e-3, "perfect rollout to cause should be ~0, got {lp}");
    }

    #[test]
    fn test_rollout_supervision_gradient_flows() {
        let r = Var::new(rollout_with_query_row(3, 2, &[0.2, 0.3, 0.5]), true);
        rollout_supervision_loss(&r, &[(0, 2, 0)]).unwrap().backward();
        assert!(r.grad().is_some());
    }

    /// Build a [heads, seq, seq] attention tensor whose query-row `q` is
    /// `row`, with every other row uniform.
    fn attn_with_query_row(heads: usize, seq: usize, q: usize, row: &[f32]) -> Tensor {
        let uniform = 1.0 / seq as f32;
        let mut data = vec![uniform; heads * seq * seq];
        for h in 0..heads {
            for j in 0..seq {
                data[h * seq * seq + q * seq + j] = row[j];
            }
        }
        Tensor::new(data, vec![heads, seq, seq])
    }

    #[test]
    fn test_attribution_none_when_empty() {
        let attn = vec![Var::new(Tensor::zeros(&[2, 3, 3]), false)];
        assert!(attribution_supervision_loss(&attn, &[], 2).is_none());
    }

    #[test]
    fn test_attribution_lower_when_pointing_at_cause() {
        // Symptom query = 2, cause = 0.
        let pointed = attn_with_query_row(2, 3, 2, &[1.0, 0.0, 0.0]); // all mass on cause
        let uniform = attn_with_query_row(2, 3, 2, &[1.0 / 3.0; 3]); // spread out

        let lp = attribution_supervision_loss(&[Var::new(pointed, false)], &[(0, 2, 0)], 2)
            .unwrap()
            .tensor()
            .data[0];
        let lu = attribution_supervision_loss(&[Var::new(uniform, false)], &[(0, 2, 0)], 2)
            .unwrap()
            .tensor()
            .data[0];
        assert!(lp < lu, "pointing at the cause should give lower loss: {lp} vs {lu}");
        assert!(lp < 1e-3, "perfect attention to cause should be ~0, got {lp}");
    }

    #[test]
    fn test_attribution_gradient_flows() {
        let attn = Var::new(attn_with_query_row(2, 3, 2, &[0.2, 0.3, 0.5]), true);
        let loss = attribution_supervision_loss(&[attn.clone()], &[(0, 2, 0)], 2).unwrap();
        loss.backward();
        assert!(attn.grad().is_some(), "attention should receive gradient");
    }

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
