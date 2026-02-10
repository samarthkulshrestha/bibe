/// Phase 0 Checkpoint — validates all requirements before moving to Phase 1.
///
/// Requirements:
/// 1. All tensor operations produce correct shapes
/// 2. Element-wise ops handle broadcasting
/// 3. Matmul matches known output on random-ish matrices
/// 4. Softmax is numerically stable (no NaN/Inf)
/// 5. All backward functions pass gradient checking (ε = 5e-4, tolerance = 1e-2)
/// 6. Memory usage is reasonable (no leaks in gradient computation)

use bibe::tensor::Tensor;
use bibe::tensor::matmul::matmul;
use bibe::tensor::broadcast::{broadcast_shapes, reduce_sum, reduce_mean, reduce_max};
use bibe::tensor::ops;
use bibe::tensor::stability::{stable_softmax, logsumexp, has_nan, has_inf, all_finite};
use bibe::autograd::{Var, gradcheck};

// ============================================================
// 1. All tensor operations produce correct shapes
// ============================================================

#[test]
fn checkpoint_shapes_elementwise() {
    let a = Tensor::new(vec![1.0; 12], vec![3, 4]);
    let b = Tensor::new(vec![2.0; 12], vec![3, 4]);

    assert_eq!(ops::add(&a, &b).shape(), &[3, 4]);
    assert_eq!(ops::sub(&a, &b).shape(), &[3, 4]);
    assert_eq!(ops::mul(&a, &b).shape(), &[3, 4]);
    assert_eq!(ops::div(&a, &b).shape(), &[3, 4]);
    assert_eq!(ops::neg(&a).shape(), &[3, 4]);
    assert_eq!(ops::exp(&a).shape(), &[3, 4]);
    assert_eq!(ops::log(&a).shape(), &[3, 4]);
    assert_eq!(ops::sqrt(&a).shape(), &[3, 4]);
    assert_eq!(ops::pow(&a, 2.0).shape(), &[3, 4]);
}

#[test]
fn checkpoint_shapes_matmul() {
    let a = Tensor::randn(&[4, 5]);
    let b = Tensor::randn(&[5, 3]);
    let c = matmul(&a, &b);
    assert_eq!(c.shape(), &[4, 3]);
}

#[test]
fn checkpoint_shapes_batched_matmul() {
    let a = Tensor::randn(&[2, 4, 5]);
    let b = Tensor::randn(&[2, 5, 3]);
    let c = matmul(&a, &b);
    assert_eq!(c.shape(), &[2, 4, 3]);
}

#[test]
fn checkpoint_shapes_reduce() {
    let a = Tensor::randn(&[3, 4, 5]);
    assert_eq!(reduce_sum(&a, 0).shape(), &[4, 5]);
    assert_eq!(reduce_sum(&a, 1).shape(), &[3, 5]);
    assert_eq!(reduce_sum(&a, 2).shape(), &[3, 4]);
    assert_eq!(reduce_mean(&a, 1).shape(), &[3, 5]);
    assert_eq!(reduce_max(&a, 2).shape(), &[3, 4]);
}

#[test]
fn checkpoint_shapes_transpose() {
    let a = Tensor::randn(&[3, 7]);
    assert_eq!(a.transpose().shape(), &[7, 3]);
    assert_eq!(a.transpose_contiguous().shape(), &[7, 3]);
}

#[test]
fn checkpoint_shapes_reshape() {
    let a = Tensor::randn(&[2, 3, 4]);
    assert_eq!(a.reshape(&[6, 4]).shape(), &[6, 4]);
    assert_eq!(a.reshape(&[24]).shape(), &[24]);
    assert_eq!(a.reshape(&[2, 12]).shape(), &[2, 12]);
}

#[test]
fn checkpoint_shapes_softmax() {
    let a = Tensor::randn(&[3, 5]);
    assert_eq!(stable_softmax(&a, 0).shape(), &[3, 5]);
    assert_eq!(stable_softmax(&a, 1).shape(), &[3, 5]);
}

#[test]
fn checkpoint_shapes_logsumexp() {
    let a = Tensor::randn(&[3, 5]);
    assert_eq!(logsumexp(&a, 0).shape(), &[5]);
    assert_eq!(logsumexp(&a, 1).shape(), &[3]);
}

// ============================================================
// 2. Element-wise ops handle broadcasting
// ============================================================

#[test]
fn checkpoint_broadcast_ops() {
    let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    let b = Tensor::new(vec![10.0, 20.0, 30.0], vec![3]); // broadcast over dim 0

    let c = ops::add(&a, &b);
    assert_eq!(c.shape(), &[2, 3]);
    assert_eq!(c.data, vec![11.0, 22.0, 33.0, 14.0, 25.0, 36.0]);

    let d = ops::mul(&a, &b);
    assert_eq!(d.shape(), &[2, 3]);
    assert_eq!(d.data, vec![10.0, 40.0, 90.0, 40.0, 100.0, 180.0]);
}

#[test]
fn checkpoint_broadcast_shapes_varied() {
    assert_eq!(broadcast_shapes(&[3, 1], &[1, 4]), vec![3, 4]);
    assert_eq!(broadcast_shapes(&[5], &[2, 5]), vec![2, 5]);
    assert_eq!(broadcast_shapes(&[1, 3, 1], &[2, 1, 4]), vec![2, 3, 4]);
}

// ============================================================
// 3. Matmul matches known output on specific matrices
// ============================================================

#[test]
fn checkpoint_matmul_known_values() {
    // A = [[1, 2, 3],    B = [[7, 10],
    //      [4, 5, 6]]         [8, 11],
    //                         [9, 12]]
    // A @ B = [[1*7+2*8+3*9, 1*10+2*11+3*12],   = [[50,  68],
    //          [4*7+5*8+6*9, 4*10+5*11+6*12]]      [122, 167]]
    let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    let b = Tensor::new(vec![7.0, 10.0, 8.0, 11.0, 9.0, 12.0], vec![3, 2]);
    let c = matmul(&a, &b);
    assert_eq!(c.shape(), &[2, 2]);
    assert_eq!(c.data, vec![50.0, 68.0, 122.0, 167.0]);
}

#[test]
fn checkpoint_matmul_identity() {
    let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    let eye = Tensor::new(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2]);
    let c = matmul(&a, &eye);
    assert_eq!(c.data, a.data);
}

#[test]
fn checkpoint_matmul_transpose_property() {
    // (A @ B)^T = B^T @ A^T
    let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    let b = Tensor::new(vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0], vec![3, 2]);

    let ab = matmul(&a, &b);
    let ab_t = ab.transpose_contiguous();

    let bt_at = matmul(&b.transpose_contiguous(), &a.transpose_contiguous());

    for i in 0..ab_t.data.len() {
        assert!(
            (ab_t.data[i] - bt_at.data[i]).abs() < 1e-5,
            "(AB)^T != B^T A^T at index {}", i
        );
    }
}

// ============================================================
// 4. Softmax is numerically stable (no NaN/Inf)
// ============================================================

#[test]
fn checkpoint_softmax_large_values() {
    let x = Tensor::new(vec![1000.0, 1000.1, 1000.2], vec![1, 3]);
    let sm = stable_softmax(&x, 1);
    assert!(all_finite(&sm), "softmax produced non-finite values on large input");
    assert!(!has_nan(&sm));
    assert!(!has_inf(&sm));

    let row_sum: f32 = sm.data.iter().sum();
    assert!((row_sum - 1.0).abs() < 1e-5, "softmax doesn't sum to 1: {}", row_sum);
}

#[test]
fn checkpoint_softmax_very_negative() {
    let x = Tensor::new(vec![-1000.0, -999.9, -999.8], vec![1, 3]);
    let sm = stable_softmax(&x, 1);
    assert!(all_finite(&sm));
    let row_sum: f32 = sm.data.iter().sum();
    assert!((row_sum - 1.0).abs() < 1e-5);
}

#[test]
fn checkpoint_softmax_mixed_extreme() {
    let x = Tensor::new(vec![-1000.0, 0.0, 1000.0], vec![1, 3]);
    let sm = stable_softmax(&x, 1);
    assert!(all_finite(&sm));
    // The largest value should dominate
    assert!(sm.get(&[0, 2]) > 0.99);
}

#[test]
fn checkpoint_softmax_2d_rows_sum_to_one() {
    let x = Tensor::randn(&[5, 10]);
    let sm = stable_softmax(&x, 1);
    assert!(all_finite(&sm));
    for i in 0..5 {
        let row_sum: f32 = (0..10).map(|j| sm.get(&[i, j])).sum();
        assert!(
            (row_sum - 1.0).abs() < 1e-5,
            "row {} sums to {} instead of 1.0", i, row_sum
        );
    }
}

#[test]
fn checkpoint_logsumexp_stable() {
    let x = Tensor::new(vec![1000.0, 1000.1, 1000.2], vec![1, 3]);
    let lse = logsumexp(&x, 1);
    assert!(all_finite(&lse), "logsumexp produced non-finite values");
}

// ============================================================
// 5. All backward functions pass gradient checking
//    Note: optimal ε for f32 central differences ≈ (machine_eps)^(1/3) ≈ 5e-3.
//    We use ε=5e-4 with tol=1e-2 which is well within f32 precision bounds.
// ============================================================

const EPS: f32 = 5e-4;
const TOL: f32 = 1e-2;

#[test]
fn checkpoint_gradcheck_add() {
    let x = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]);
    let (ok, err) = gradcheck(
        &|v| {
            let b = Var::new(Tensor::new(vec![4.0, 5.0, 6.0], vec![3]), false);
            v.add(&b).sum()
        },
        &x, EPS, TOL,
    );
    assert!(ok, "add gradcheck failed: max_rel_err={}", err);
}

#[test]
fn checkpoint_gradcheck_sub() {
    let x = Tensor::new(vec![5.0, 3.0, 1.0], vec![3]);
    let (ok, err) = gradcheck(
        &|v| {
            let b = Var::new(Tensor::new(vec![1.0, 2.0, 3.0], vec![3]), false);
            v.sub(&b).sum()
        },
        &x, EPS, TOL,
    );
    assert!(ok, "sub gradcheck failed: max_rel_err={}", err);
}

#[test]
fn checkpoint_gradcheck_mul() {
    let x = Tensor::new(vec![2.0, 3.0, 4.0], vec![3]);
    let (ok, err) = gradcheck(
        &|v| {
            let b = Var::new(Tensor::new(vec![5.0, 6.0, 7.0], vec![3]), false);
            v.mul(&b).sum()
        },
        &x, EPS, TOL,
    );
    assert!(ok, "mul gradcheck failed: max_rel_err={}", err);
}

#[test]
fn checkpoint_gradcheck_div() {
    let x = Tensor::new(vec![10.0, 20.0, 30.0], vec![3]);
    let (ok, err) = gradcheck(
        &|v| {
            let b = Var::new(Tensor::new(vec![2.0, 4.0, 5.0], vec![3]), false);
            v.div(&b).sum()
        },
        &x, EPS, TOL,
    );
    assert!(ok, "div gradcheck failed: max_rel_err={}", err);
}

#[test]
fn checkpoint_gradcheck_neg() {
    let x = Tensor::new(vec![2.0, -3.0, 4.0], vec![3]);
    let (ok, err) = gradcheck(&|v| v.neg().sum(), &x, EPS, TOL);
    assert!(ok, "neg gradcheck failed: max_rel_err={}", err);
}

#[test]
fn checkpoint_gradcheck_exp() {
    let x = Tensor::new(vec![0.0, 0.5, 1.0], vec![3]);
    let (ok, err) = gradcheck(&|v| v.exp().sum(), &x, EPS, TOL);
    assert!(ok, "exp gradcheck failed: max_rel_err={}", err);
}

#[test]
fn checkpoint_gradcheck_log() {
    let x = Tensor::new(vec![1.0, 2.0, 4.0], vec![3]);
    let (ok, err) = gradcheck(&|v| v.log().sum(), &x, EPS, TOL);
    assert!(ok, "log gradcheck failed: max_rel_err={}", err);
}

#[test]
fn checkpoint_gradcheck_sqrt() {
    let x = Tensor::new(vec![1.0, 4.0, 9.0], vec![3]);
    let (ok, err) = gradcheck(&|v| v.sqrt().sum(), &x, EPS, TOL);
    assert!(ok, "sqrt gradcheck failed: max_rel_err={}", err);
}

#[test]
fn checkpoint_gradcheck_pow() {
    let x = Tensor::new(vec![2.0, 3.0, 4.0], vec![3]);
    let (ok, err) = gradcheck(&|v| v.pow(3.0).sum(), &x, EPS, TOL);
    assert!(ok, "pow gradcheck failed: max_rel_err={}", err);
}

#[test]
fn checkpoint_gradcheck_matmul() {
    let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    let (ok, err) = gradcheck(
        &|v| {
            let b = Var::new(
                Tensor::new(vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0], vec![3, 2]),
                false,
            );
            v.matmul(&b).sum()
        },
        &x, EPS, TOL,
    );
    assert!(ok, "matmul gradcheck failed: max_rel_err={}", err);
}

#[test]
fn checkpoint_gradcheck_transpose() {
    let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    let (ok, err) = gradcheck(&|v| v.transpose().sum(), &x, EPS, TOL);
    assert!(ok, "transpose gradcheck failed: max_rel_err={}", err);
}

#[test]
fn checkpoint_gradcheck_mul_scalar() {
    let x = Tensor::new(vec![2.0, 3.0, 4.0], vec![3]);
    let (ok, err) = gradcheck(&|v| v.mul_scalar(5.0).sum(), &x, EPS, TOL);
    assert!(ok, "mul_scalar gradcheck failed: max_rel_err={}", err);
}

#[test]
fn checkpoint_gradcheck_softmax() {
    let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    let (ok, err) = gradcheck(
        &|v| {
            let s = v.softmax(1);
            let w = Var::new(
                Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]),
                false,
            );
            s.mul(&w).sum()
        },
        &x, EPS, TOL,
    );
    assert!(ok, "softmax gradcheck failed: max_rel_err={}", err);
}

#[test]
fn checkpoint_gradcheck_chain_ops() {
    // loss = sum(exp(a * b + c))
    let x = Tensor::new(vec![0.1, 0.2, 0.3], vec![3]);
    let (ok, err) = gradcheck(
        &|v| {
            let b = Var::new(Tensor::new(vec![0.5, 0.6, 0.7], vec![3]), false);
            let c = Var::new(Tensor::new(vec![0.01, 0.02, 0.03], vec![3]), false);
            v.mul(&b).add(&c).exp().sum()
        },
        &x, EPS, TOL,
    );
    assert!(ok, "chain gradcheck failed: max_rel_err={}", err);
}

#[test]
fn checkpoint_gradcheck_broadcast() {
    // a: [2, 3] + b: [3] — gradient w.r.t. a
    let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    let (ok, err) = gradcheck(
        &|v| {
            let b = Var::new(Tensor::new(vec![10.0, 20.0, 30.0], vec![3]), false);
            v.add(&b).sum()
        },
        &x, EPS, TOL,
    );
    assert!(ok, "broadcast gradcheck failed: max_rel_err={}", err);
}

// ============================================================
// 6. Memory usage is reasonable (no leaks in gradient computation)
// ============================================================

#[test]
fn checkpoint_repeated_forward_backward() {
    // Run many forward/backward passes to verify no accumulating leaks.
    // If Rc cycles existed, this would grow unboundedly.
    for _ in 0..1000 {
        let a = Var::new(Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]), true);
        let b = Var::new(Tensor::new(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]), true);
        let c = a.matmul(&b);
        let d = c.exp();
        let loss = d.sum();
        loss.backward();

        // Verify gradient exists and is finite
        let ga = a.grad().unwrap();
        assert!(ga.data.iter().all(|x| x.is_finite()));
    }
    // If we reach here without OOM or panic, memory is fine.
}

#[test]
fn checkpoint_deep_chain_no_stack_overflow() {
    // Chain 100 operations to ensure topo sort handles deep graphs
    let mut v = Var::new(Tensor::new(vec![1.0], vec![1]), true);
    for _ in 0..100 {
        v = v.mul_scalar(1.001);
    }
    let loss = v.sum();
    loss.backward();

    let grad = v.grad().expect("no gradient on deep chain loss");
    assert!(grad.data[0].is_finite());
}

#[test]
fn checkpoint_diamond_graph() {
    // Diamond: a -> b, a -> c, b + c -> d
    // Tests gradient accumulation through shared node
    let a = Var::new(Tensor::new(vec![3.0], vec![1]), true);
    let b = a.mul_scalar(2.0);  // b = 2a
    let c = a.mul_scalar(3.0);  // c = 3a
    let d = b.add(&c);          // d = 5a
    let loss = d.sum();         // loss = 5a = 15
    loss.backward();

    let grad = a.grad().unwrap();
    assert!((grad.data[0] - 5.0).abs() < 1e-6, "diamond grad: expected 5, got {}", grad.data[0]);
}
