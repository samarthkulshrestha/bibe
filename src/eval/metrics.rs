//! Quantitative evaluation metrics.
//!
//! Anomaly detection is scored with AUC-ROC and Precision@K over per-event
//! scores and binary labels; attribution/localization is scored with Hit@K
//! and Mean Reciprocal Rank over events ranked by score against a known
//! ground-truth index.

/// Indices `0..scores.len()` ordered by score, highest first (stable).
pub fn rank_by_score_desc(scores: &[f32]) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..scores.len()).collect();
    // Stable sort by descending score; ties keep original order.
    idx.sort_by(|&a, &b| scores[b].partial_cmp(&scores[a]).unwrap());
    idx
}

/// Area under the ROC curve via the rank (Mann-Whitney U) statistic, with
/// average ranks for ties. Returns 0.5 when one class is absent.
pub fn auc_roc(scores: &[f32], labels: &[bool]) -> f32 {
    let n = scores.len();
    let n_pos = labels.iter().filter(|&&l| l).count();
    let n_neg = n - n_pos;
    if n_pos == 0 || n_neg == 0 {
        return 0.5;
    }

    // Indices sorted ascending by score.
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&a, &b| scores[a].partial_cmp(&scores[b]).unwrap());

    // Average ranks (1-based), tied scores share the mean of their ranks.
    let mut ranks = vec![0.0f32; n];
    let mut i = 0;
    while i < n {
        let mut j = i;
        while j + 1 < n && scores[idx[j + 1]] == scores[idx[i]] {
            j += 1;
        }
        let avg_rank = ((i + 1) + (j + 1)) as f32 / 2.0;
        for &orig in &idx[i..=j] {
            ranks[orig] = avg_rank;
        }
        i = j + 1;
    }

    let sum_pos: f32 = (0..n).filter(|&k| labels[k]).map(|k| ranks[k]).sum();
    let n_pos = n_pos as f32;
    let n_neg = n_neg as f32;
    (sum_pos - n_pos * (n_pos + 1.0) / 2.0) / (n_pos * n_neg)
}

/// Fraction of the top-`k` highest-scoring items that are positive.
pub fn precision_at_k(scores: &[f32], labels: &[bool], k: usize) -> f32 {
    let k = k.min(scores.len());
    if k == 0 {
        return 0.0;
    }
    let ranked = rank_by_score_desc(scores);
    let hits = ranked[..k].iter().filter(|&&i| labels[i]).count();
    hits as f32 / k as f32
}

/// Whether `ground_truth` appears within the top `k` of a ranking.
pub fn hit_at_k(ranked: &[usize], ground_truth: usize, k: usize) -> bool {
    let k = k.min(ranked.len());
    ranked[..k].contains(&ground_truth)
}

/// Reciprocal of the 1-based position of `ground_truth` in `ranked`
/// (0.0 if absent).
pub fn mrr(ranked: &[usize], ground_truth: usize) -> f32 {
    match ranked.iter().position(|&i| i == ground_truth) {
        Some(pos) => 1.0 / (pos + 1) as f32,
        None => 0.0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rank_orders_high_to_low() {
        let r = rank_by_score_desc(&[0.1, 0.9, 0.3, 0.8]);
        assert_eq!(r, vec![1, 3, 2, 0]);
    }

    #[test]
    fn test_auc_perfect_separation() {
        let scores = [0.1, 0.4, 0.35, 0.8];
        let labels = [false, true, false, true];
        assert!((auc_roc(&scores, &labels) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_auc_inverted_is_zero() {
        let scores = [0.1, 0.4, 0.35, 0.8];
        let labels = [true, false, true, false];
        assert!(auc_roc(&scores, &labels).abs() < 1e-6);
    }

    #[test]
    fn test_auc_all_tied_is_half() {
        let scores = [0.5, 0.5, 0.5, 0.5];
        let labels = [true, false, true, false];
        assert!((auc_roc(&scores, &labels) - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_auc_single_class_is_half() {
        assert!((auc_roc(&[0.1, 0.9], &[true, true]) - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_precision_at_k() {
        let scores = [0.9, 0.1, 0.8, 0.2];
        let labels = [true, false, true, false];
        assert!((precision_at_k(&scores, &labels, 2) - 1.0).abs() < 1e-6);
        assert!((precision_at_k(&scores, &labels, 4) - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_precision_at_k_clamps_k() {
        let scores = [0.9, 0.1];
        let labels = [true, false];
        // k beyond length is clamped to the length.
        assert!((precision_at_k(&scores, &labels, 10) - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_hit_at_k() {
        let ranked = [3, 1, 2, 0];
        assert!(hit_at_k(&ranked, 1, 2));
        assert!(!hit_at_k(&ranked, 1, 1));
        assert!(!hit_at_k(&ranked, 7, 4));
    }

    #[test]
    fn test_mrr() {
        let ranked = [3, 1, 2, 0];
        assert!((mrr(&ranked, 3) - 1.0).abs() < 1e-6); // first
        assert!((mrr(&ranked, 2) - 1.0 / 3.0).abs() < 1e-6); // third
        assert!(mrr(&ranked, 9).abs() < 1e-6); // absent
    }
}
