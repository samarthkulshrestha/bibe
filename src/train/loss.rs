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

/// Mean focal loss for binary classification (Lin et al., 2017).
///
/// ```text
/// FL = -mean[ α·(1-p)^γ·y·log(p) + (1-α)·p^γ·(1-y)·log(1-p) ]
/// ```
///
/// The `(1-p)^γ` / `p^γ` modulating factors down-weight easy, well-classified
/// examples so training focuses on hard ones — useful for the heavy class
/// imbalance in anomaly detection. Typical settings are `alpha = 0.75`,
/// `gamma = 2.0`.
pub fn focal_loss(pred: &Var, target: &Var, alpha: f32, gamma: f32) -> Var {
    let n = pred.tensor().data.len() as f32;
    let p = pred.clamp(EPS, 1.0 - EPS);
    let one_minus_p = p.neg().add_scalar(1.0);
    let one_minus_y = target.neg().add_scalar(1.0);

    // α·(1-p)^γ·y·log(p)
    let pos = one_minus_p
        .pow(gamma)
        .mul(target)
        .mul(&p.log())
        .mul_scalar(alpha);
    // (1-α)·p^γ·(1-y)·log(1-p)
    let neg = p
        .pow(gamma)
        .mul(&one_minus_y)
        .mul(&one_minus_p.log())
        .mul_scalar(1.0 - alpha);

    pos.add(&neg).sum().mul_scalar(-1.0 / n)
}

/// Cosine similarity between two equal-length vectors, as a scalar `Var`.
fn cosine_similarity(a: &Var, b: &Var) -> Var {
    let dot = a.mul(b).sum();
    let norm_a = a.mul(a).sum().sqrt();
    let norm_b = b.mul(b).sum().sqrt();
    dot.div(&norm_a.mul(&norm_b))
}

/// Contrastive trace loss (InfoNCE) over cosine similarities.
///
/// ```text
/// L = -log( exp(sim(a, pos)/τ) / (exp(sim(a, pos)/τ) + Σ_n exp(sim(a, neg_n)/τ)) )
/// ```
///
/// Pulls the anchor's representation toward the positive and away from the
/// negatives at temperature `temperature` (e.g. 0.07). Cosine similarity is
/// bounded in `[-1, 1]`, so the scaled logits cannot overflow `exp` and no
/// max-subtraction is needed.
pub fn contrastive_loss(
    anchor: &Var,
    positive: &Var,
    negatives: &[Var],
    temperature: f32,
) -> Var {
    let inv_t = 1.0 / temperature;
    let pos_logit = cosine_similarity(anchor, positive).mul_scalar(inv_t);

    // denom = exp(pos) + Σ_n exp(neg_n)
    let mut denom = pos_logit.exp();
    for neg in negatives {
        let neg_logit = cosine_similarity(anchor, neg).mul_scalar(inv_t);
        denom = denom.add(&neg_logit.exp());
    }

    // -log(exp(pos)/denom) = log(denom) - pos
    denom.log().sub(&pos_logit)
}

/// Attention sparsity loss: the (scaled) Shannon entropy of the attention
/// weights.
///
/// ```text
/// L_sparse = λ · Σ_{i,j} -A[i,j]·log(A[i,j])
/// ```
///
/// Minimizing entropy encourages each query to concentrate on a few source
/// events rather than spreading attention uniformly, which aids causal
/// attribution. A typical weight is `lambda = 0.01`.
pub fn attention_sparsity_loss(attn: &Var, lambda: f32) -> Var {
    let a = attn.clamp(EPS, 1.0);
    // Σ -p·log(p), scaled by lambda.
    a.mul(&a.log()).sum().mul_scalar(-lambda)
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

    #[test]
    fn test_focal_known_value() {
        // p=0.5, y=1, alpha=0.75, gamma=2: -0.75*(0.5)^2*ln(0.5) ≈ 0.12996
        let pred = Var::new(Tensor::new(vec![0.5], vec![1]), false);
        let target = Var::new(Tensor::new(vec![1.0], vec![1]), false);
        let loss = focal_loss(&pred, &target, 0.75, 2.0);
        let expected = -0.75 * 0.25 * 0.5_f32.ln();
        assert!((loss.tensor().data[0] - expected).abs() < 1e-5);
    }

    #[test]
    fn test_focal_downweights_easy_examples() {
        // For a well-classified positive, focal loss is far below plain BCE.
        let pred = Var::new(Tensor::new(vec![0.9], vec![1]), false);
        let target = Var::new(Tensor::new(vec![1.0], vec![1]), false);
        let focal = focal_loss(&pred, &target, 0.75, 2.0).tensor().data[0];
        let bce = bce_loss(&pred, &target).tensor().data[0];
        assert!(focal < bce, "focal {focal} should be < bce {bce} on an easy example");
    }

    #[test]
    fn test_focal_gradient_numeric() {
        let pred = Tensor::new(vec![0.6, 0.3, 0.8], vec![3]);
        let target = Var::new(Tensor::new(vec![1.0, 0.0, 1.0], vec![3]), false);
        let (ok, err) =
            gradcheck(&|p: &Var| focal_loss(p, &target, 0.75, 2.0), &pred, 1e-3, 3e-3);
        assert!(ok, "focal gradcheck failed, max rel err = {err}");
    }

    #[test]
    fn test_contrastive_known_value() {
        // τ=1, sim(a,pos)=1, sim(a,neg)=0 -> log(e + 1) - 1 ≈ 0.31326.
        let anchor = Var::new(Tensor::new(vec![1.0, 0.0], vec![2]), false);
        let positive = Var::new(Tensor::new(vec![1.0, 0.0], vec![2]), false);
        let negatives = vec![Var::new(Tensor::new(vec![0.0, 1.0], vec![2]), false)];
        let loss = contrastive_loss(&anchor, &positive, &negatives, 1.0);
        let expected = (std::f32::consts::E + 1.0).ln() - 1.0;
        assert!((loss.tensor().data[0] - expected).abs() < 1e-5);
    }

    #[test]
    fn test_contrastive_lower_when_positive_aligned() {
        let anchor = Var::new(Tensor::new(vec![1.0, 0.0], vec![2]), false);
        let neg = Var::new(Tensor::new(vec![0.0, 1.0], vec![2]), false);

        // Positive aligned with anchor, negative orthogonal -> low loss.
        let aligned = contrastive_loss(
            &anchor,
            &Var::new(Tensor::new(vec![1.0, 0.0], vec![2]), false),
            &[neg.clone()],
            0.5,
        );
        // Positive orthogonal, negative aligned with anchor -> high loss.
        let misaligned = contrastive_loss(
            &anchor,
            &Var::new(Tensor::new(vec![0.0, 1.0], vec![2]), false),
            &[Var::new(Tensor::new(vec![1.0, 0.0], vec![2]), false)],
            0.5,
        );
        assert!(
            aligned.tensor().data[0] < misaligned.tensor().data[0],
            "aligned positive should give lower loss"
        );
    }

    #[test]
    fn test_contrastive_gradient_numeric() {
        let anchor = Tensor::new(vec![0.5, 0.3], vec![2]);
        let positive = Var::new(Tensor::new(vec![1.0, 0.0], vec![2]), false);
        let negatives = vec![Var::new(Tensor::new(vec![0.0, 1.0], vec![2]), false)];
        let (ok, err) = gradcheck(
            &|a: &Var| contrastive_loss(a, &positive, &negatives, 1.0),
            &anchor,
            1e-3,
            3e-3,
        );
        assert!(ok, "contrastive gradcheck failed, max rel err = {err}");
    }

    #[test]
    fn test_sparsity_one_hot_is_zero() {
        // One-hot rows have zero entropy.
        let attn = Var::new(Tensor::new(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2]), false);
        let loss = attention_sparsity_loss(&attn, 1.0);
        assert!(loss.tensor().data[0] < 1e-3, "one-hot entropy should be ~0");
    }

    #[test]
    fn test_sparsity_uniform_is_max() {
        // Two uniform rows of width 2: total entropy = 2·ln 2.
        let attn = Var::new(Tensor::new(vec![0.5, 0.5, 0.5, 0.5], vec![2, 2]), false);
        let loss = attention_sparsity_loss(&attn, 1.0);
        assert!((loss.tensor().data[0] - 2.0 * 2.0_f32.ln()).abs() < 1e-4);
    }

    #[test]
    fn test_sparsity_penalizes_uniform_more_than_peaked() {
        let uniform = Var::new(Tensor::new(vec![0.5, 0.5], vec![1, 2]), false);
        let peaked = Var::new(Tensor::new(vec![0.95, 0.05], vec![1, 2]), false);
        let lu = attention_sparsity_loss(&uniform, 1.0).tensor().data[0];
        let lp = attention_sparsity_loss(&peaked, 1.0).tensor().data[0];
        assert!(lp < lu, "peaked attention {lp} should be penalized less than uniform {lu}");
    }

    #[test]
    fn test_sparsity_gradient_numeric() {
        let attn = Tensor::new(vec![0.3, 0.7, 0.6, 0.4], vec![2, 2]);
        let (ok, err) = gradcheck(&|a: &Var| attention_sparsity_loss(a, 1.0), &attn, 1e-3, 2e-3);
        assert!(ok, "sparsity gradcheck failed, max rel err = {err}");
    }
}
