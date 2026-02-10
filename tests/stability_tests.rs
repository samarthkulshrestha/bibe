use bibe::tensor::Tensor;
use bibe::tensor::stability::{
    stable_softmax, logsumexp, clip, safe_log,
    has_nan, has_inf, all_finite,
};

fn assert_approx_eq(a: f32, b: f32, tol: f32) {
    assert!(
        (a - b).abs() < tol,
        "{} vs {} (diff {})",
        a, b, (a - b).abs()
    );
}

// ============================================================
// stable_softmax
// ============================================================

#[test]
fn test_softmax_sums_to_one() {
    let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    let sm = stable_softmax(&x, 1);

    assert_eq!(sm.shape(), &[2, 3]);
    for i in 0..2 {
        let row_sum: f32 = (0..3).map(|j| sm.get(&[i, j])).sum();
        assert_approx_eq(row_sum, 1.0, 1e-6);
    }
}

#[test]
fn test_softmax_sums_to_one_dim0() {
    let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    let sm = stable_softmax(&x, 0);

    assert_eq!(sm.shape(), &[2, 3]);
    for j in 0..3 {
        let col_sum: f32 = (0..2).map(|i| sm.get(&[i, j])).sum();
        assert_approx_eq(col_sum, 1.0, 1e-6);
    }
}

#[test]
fn test_softmax_all_positive() {
    let x = Tensor::new(vec![-5.0, 0.0, 5.0, -100.0, 0.0, 100.0], vec![2, 3]);
    let sm = stable_softmax(&x, 1);

    for i in 0..2 {
        for j in 0..3 {
            assert!(sm.get(&[i, j]) >= 0.0, "softmax output must be non-negative");
            assert!(sm.get(&[i, j]) <= 1.0, "softmax output must be <= 1.0");
        }
    }
}

#[test]
fn test_softmax_large_values_no_overflow() {
    let x = Tensor::new(vec![1000.0, 1000.1, 1000.2], vec![1, 3]);
    let sm = stable_softmax(&x, 1);

    assert!(all_finite(&sm), "softmax on large values should not produce NaN/Inf");
    let sum: f32 = (0..3).map(|j| sm.get(&[0, j])).sum();
    assert_approx_eq(sum, 1.0, 1e-6);
}

#[test]
fn test_softmax_very_negative_values_no_underflow() {
    let x = Tensor::new(vec![-1000.0, -1000.1, -1000.2], vec![1, 3]);
    let sm = stable_softmax(&x, 1);

    assert!(all_finite(&sm), "softmax on very negative values should not produce NaN/Inf");
    let sum: f32 = (0..3).map(|j| sm.get(&[0, j])).sum();
    assert_approx_eq(sum, 1.0, 1e-6);
}

#[test]
fn test_softmax_equal_inputs() {
    // Equal inputs should give uniform distribution
    let x = Tensor::new(vec![5.0, 5.0, 5.0, 5.0], vec![1, 4]);
    let sm = stable_softmax(&x, 1);

    for j in 0..4 {
        assert_approx_eq(sm.get(&[0, j]), 0.25, 1e-6);
    }
}

#[test]
fn test_softmax_known_values() {
    // softmax([0, 1, 2]) = [e^0, e^1, e^2] / (e^0 + e^1 + e^2)
    let x = Tensor::new(vec![0.0, 1.0, 2.0], vec![1, 3]);
    let sm = stable_softmax(&x, 1);

    let e0 = 1.0_f32;
    let e1 = 1.0_f32.exp();
    let e2 = 2.0_f32.exp();
    let total = e0 + e1 + e2;

    assert_approx_eq(sm.get(&[0, 0]), e0 / total, 1e-6);
    assert_approx_eq(sm.get(&[0, 1]), e1 / total, 1e-6);
    assert_approx_eq(sm.get(&[0, 2]), e2 / total, 1e-6);
}

#[test]
fn test_softmax_preserves_ordering() {
    let x = Tensor::new(vec![1.0, 3.0, 2.0], vec![1, 3]);
    let sm = stable_softmax(&x, 1);

    assert!(sm.get(&[0, 1]) > sm.get(&[0, 2]));
    assert!(sm.get(&[0, 2]) > sm.get(&[0, 0]));
}

#[test]
fn test_softmax_3d() {
    // [2, 2, 3], softmax along last dim
    let x = Tensor::new(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
        vec![2, 2, 3],
    );
    let sm = stable_softmax(&x, 2);

    assert_eq!(sm.shape(), &[2, 2, 3]);
    for i in 0..2 {
        for j in 0..2 {
            let sum: f32 = (0..3).map(|k| sm.get(&[i, j, k])).sum();
            assert_approx_eq(sum, 1.0, 1e-5);
        }
    }
}

#[test]
fn test_softmax_method() {
    let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    let sm1 = stable_softmax(&x, 1);
    let sm2 = x.softmax(1);
    assert_eq!(sm1.data, sm2.data);
}

// ============================================================
// logsumexp
// ============================================================

#[test]
fn test_logsumexp_matches_naive_on_safe_inputs() {
    let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    let lse = logsumexp(&x, 1);

    assert_eq!(lse.shape(), &[2]);

    // Row 0: log(e^1 + e^2 + e^3)
    let expected0 = (1.0_f32.exp() + 2.0_f32.exp() + 3.0_f32.exp()).ln();
    // Row 1: log(e^4 + e^5 + e^6)
    let expected1 = (4.0_f32.exp() + 5.0_f32.exp() + 6.0_f32.exp()).ln();

    assert_approx_eq(lse.get(&[0]), expected0, 1e-5);
    assert_approx_eq(lse.get(&[1]), expected1, 1e-5);
}

#[test]
fn test_logsumexp_large_values_no_overflow() {
    let x = Tensor::new(vec![1000.0, 1000.1, 1000.2], vec![1, 3]);
    let lse = logsumexp(&x, 1);

    assert!(all_finite(&lse), "logsumexp on large values should not overflow");
    // logsumexp([1000, 1000.1, 1000.2]) ≈ 1000.2 + log(e^-0.2 + e^-0.1 + 1)
    let expected = 1000.2 + ((-0.2_f32).exp() + (-0.1_f32).exp() + 1.0).ln();
    assert_approx_eq(lse.get(&[0]), expected, 1e-3);
}

#[test]
fn test_logsumexp_very_negative_values() {
    let x = Tensor::new(vec![-1000.0, -1000.1, -1000.2], vec![1, 3]);
    let lse = logsumexp(&x, 1);

    assert!(all_finite(&lse), "logsumexp on very negative values should not underflow to -Inf");
}

#[test]
fn test_logsumexp_dim0() {
    let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    let lse = logsumexp(&x, 0);

    assert_eq!(lse.shape(), &[3]);

    // Col 0: log(e^1 + e^4)
    let expected0 = (1.0_f32.exp() + 4.0_f32.exp()).ln();
    assert_approx_eq(lse.get(&[0]), expected0, 1e-5);
}

#[test]
fn test_logsumexp_single_element_per_group() {
    // logsumexp of a single element should be that element
    let x = Tensor::new(vec![3.0, 7.0], vec![2, 1]);
    let lse = logsumexp(&x, 1);

    assert_approx_eq(lse.get(&[0]), 3.0, 1e-5);
    assert_approx_eq(lse.get(&[1]), 7.0, 1e-5);
}

#[test]
fn test_logsumexp_method() {
    let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    let lse1 = logsumexp(&x, 1);
    let lse2 = x.logsumexp(1);
    assert_eq!(lse1.data, lse2.data);
}

// ============================================================
// clip
// ============================================================

#[test]
fn test_clip_basic() {
    let x = Tensor::new(vec![-5.0, -1.0, 0.0, 1.0, 5.0], vec![5]);
    let c = clip(&x, -2.0, 2.0);

    assert_eq!(c.data, vec![-2.0, -1.0, 0.0, 1.0, 2.0]);
}

#[test]
fn test_clip_all_within_range() {
    let x = Tensor::new(vec![0.1, 0.5, 0.9], vec![3]);
    let c = clip(&x, 0.0, 1.0);

    assert_eq!(c.data, vec![0.1, 0.5, 0.9]);
}

#[test]
fn test_clip_all_below() {
    let x = Tensor::new(vec![-10.0, -20.0, -30.0], vec![3]);
    let c = clip(&x, 0.0, 1.0);

    assert_eq!(c.data, vec![0.0, 0.0, 0.0]);
}

#[test]
fn test_clip_all_above() {
    let x = Tensor::new(vec![10.0, 20.0, 30.0], vec![3]);
    let c = clip(&x, 0.0, 1.0);

    assert_eq!(c.data, vec![1.0, 1.0, 1.0]);
}

#[test]
fn test_clip_preserves_shape() {
    let x = Tensor::randn(&[2, 3, 4]);
    let c = clip(&x, -1.0, 1.0);
    assert_eq!(c.shape(), &[2, 3, 4]);
}

#[test]
fn test_clip_attention_weights() {
    // Simulates clipping attention weights before log
    let x = Tensor::new(vec![0.0, 1e-10, 0.5, 1.0], vec![4]);
    let c = clip(&x, 1e-8, 1.0);

    assert_eq!(c.get(&[0]), 1e-8); // clamped up
    assert_eq!(c.get(&[1]), 1e-8); // clamped up (1e-10 < 1e-8)
    assert_eq!(c.get(&[2]), 0.5);  // unchanged
    assert_eq!(c.get(&[3]), 1.0);  // unchanged
}

#[test]
fn test_clip_method() {
    let x = Tensor::new(vec![-5.0, 0.0, 5.0], vec![3]);
    let c1 = clip(&x, -1.0, 1.0);
    let c2 = x.clip(-1.0, 1.0);
    assert_eq!(c1.data, c2.data);
}

// ============================================================
// safe_log
// ============================================================

#[test]
fn test_safe_log_normal_values() {
    let x = Tensor::new(vec![1.0, 2.718281828, 7.389056099], vec![3]);
    let l = safe_log(&x, 1e-8);

    assert_approx_eq(l.get(&[0]), 0.0, 1e-6);
    assert_approx_eq(l.get(&[1]), 1.0, 1e-6);
    assert_approx_eq(l.get(&[2]), 2.0, 1e-6);
}

#[test]
fn test_safe_log_zero_uses_epsilon() {
    let x = Tensor::new(vec![0.0], vec![1]);
    let l = safe_log(&x, 1e-8);

    // Should be ln(1e-8) ≈ -18.42
    assert!(l.get(&[0]).is_finite());
    assert_approx_eq(l.get(&[0]), 1e-8_f32.ln(), 1e-4);
}

#[test]
fn test_safe_log_negative_uses_epsilon() {
    let x = Tensor::new(vec![-1.0, -100.0], vec![2]);
    let l = safe_log(&x, 1e-8);

    assert!(all_finite(&l));
    assert_eq!(l.get(&[0]), 1e-8_f32.ln());
    assert_eq!(l.get(&[1]), 1e-8_f32.ln());
}

#[test]
fn test_safe_log_large_epsilon() {
    let x = Tensor::new(vec![0.0, 0.001, 1.0], vec![3]);
    let l = safe_log(&x, 0.01);

    // 0.0 → ln(0.01), 0.001 → ln(0.01), 1.0 → ln(1.0)
    assert_approx_eq(l.get(&[0]), 0.01_f32.ln(), 1e-6);
    assert_approx_eq(l.get(&[1]), 0.01_f32.ln(), 1e-6);
    assert_approx_eq(l.get(&[2]), 0.0, 1e-6);
}

// ============================================================
// NaN / Inf detection
// ============================================================

#[test]
fn test_has_nan_false() {
    let x = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]);
    assert!(!has_nan(&x));
}

#[test]
fn test_has_nan_true() {
    let x = Tensor::new(vec![1.0, f32::NAN, 3.0], vec![3]);
    assert!(has_nan(&x));
}

#[test]
fn test_has_inf_false() {
    let x = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]);
    assert!(!has_inf(&x));
}

#[test]
fn test_has_inf_positive() {
    let x = Tensor::new(vec![1.0, f32::INFINITY, 3.0], vec![3]);
    assert!(has_inf(&x));
}

#[test]
fn test_has_inf_negative() {
    let x = Tensor::new(vec![f32::NEG_INFINITY, 2.0, 3.0], vec![3]);
    assert!(has_inf(&x));
}

#[test]
fn test_all_finite_true() {
    let x = Tensor::new(vec![1.0, -1.0, 0.0, 1e30, -1e30], vec![5]);
    assert!(all_finite(&x));
}

#[test]
fn test_all_finite_false_nan() {
    let x = Tensor::new(vec![1.0, f32::NAN], vec![2]);
    assert!(!all_finite(&x));
}

#[test]
fn test_all_finite_false_inf() {
    let x = Tensor::new(vec![1.0, f32::INFINITY], vec![2]);
    assert!(!all_finite(&x));
}

// ============================================================
// Combined / integration
// ============================================================

#[test]
fn test_softmax_then_log_with_clip() {
    // Common pattern: log(clip(softmax(x)))
    let x = Tensor::randn(&[4, 8]);
    let sm = stable_softmax(&x, 1);
    let clipped = clip(&sm, 1e-8, 1.0);
    let log_sm = safe_log(&clipped, 1e-8);

    assert!(all_finite(&log_sm), "log(softmax) should be finite");
    // All log-softmax values should be <= 0 (since softmax in [0, 1])
    assert!(log_sm.data.iter().all(|&v| v <= 0.0));
}

#[test]
fn test_logsumexp_softmax_relationship() {
    // log_softmax(x) = x - logsumexp(x, dim, keepdim=true)
    // So: softmax(x) = exp(x - logsumexp(x, keepdim))
    let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    let sm = stable_softmax(&x, 1);
    let lse = logsumexp(&x, 1);

    // Verify relationship for each element
    for i in 0..2 {
        for j in 0..3 {
            let expected = (x.get(&[i, j]) - lse.get(&[i])).exp();
            assert_approx_eq(sm.get(&[i, j]), expected, 1e-5);
        }
    }
}

#[test]
fn test_softmax_gradient_sanity() {
    // If we increase one logit, its softmax probability should increase
    let x1 = Tensor::new(vec![1.0, 2.0, 3.0], vec![1, 3]);
    let x2 = Tensor::new(vec![1.0, 2.0, 4.0], vec![1, 3]); // increased last

    let sm1 = stable_softmax(&x1, 1);
    let sm2 = stable_softmax(&x2, 1);

    assert!(sm2.get(&[0, 2]) > sm1.get(&[0, 2]),
        "increasing logit should increase its softmax probability");
    assert!(sm2.get(&[0, 0]) < sm1.get(&[0, 0]),
        "increasing one logit should decrease others");
}
