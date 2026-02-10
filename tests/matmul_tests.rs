use bibe::tensor::Tensor;
use bibe::tensor::matmul::{matmul, matmul_blocked, batched_matmul};

// ============================================================
// Helper
// ============================================================

fn assert_approx_eq(a: &[f32], b: &[f32], tol: f32) {
    assert_eq!(a.len(), b.len(), "length mismatch: {} vs {}", a.len(), b.len());
    for (i, (&x, &y)) in a.iter().zip(b.iter()).enumerate() {
        assert!(
            (x - y).abs() < tol,
            "mismatch at index {}: {} vs {} (diff {})",
            i, x, y, (x - y).abs()
        );
    }
}

// ============================================================
// Naive 2D matmul
// ============================================================

#[test]
fn test_matmul_2d_basic() {
    // [[1, 2], [3, 4]] @ [[5, 6], [7, 8]] = [[19, 22], [43, 50]]
    let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    let b = Tensor::new(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]);
    let c = matmul(&a, &b);

    assert_eq!(c.shape(), &[2, 2]);
    assert_eq!(c.data, vec![19.0, 22.0, 43.0, 50.0]);
}

#[test]
fn test_matmul_2d_non_square() {
    // [2, 3] @ [3, 4] → [2, 4]
    let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    let b = Tensor::new(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
        vec![3, 4],
    );
    let c = matmul(&a, &b);

    assert_eq!(c.shape(), &[2, 4]);
    // row 0: [1*1+2*5+3*9, 1*2+2*6+3*10, 1*3+2*7+3*11, 1*4+2*8+3*12]
    //      = [38, 44, 50, 56]
    // row 1: [4*1+5*5+6*9, 4*2+5*6+6*10, 4*3+5*7+6*11, 4*4+5*8+6*12]
    //      = [83, 98, 113, 128]
    assert_eq!(c.data, vec![38.0, 44.0, 50.0, 56.0, 83.0, 98.0, 113.0, 128.0]);
}

#[test]
fn test_matmul_2d_identity() {
    // A @ I = A
    let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    let eye = Tensor::new(
        vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        vec![3, 3],
    );
    let c = matmul(&a, &eye);

    assert_eq!(c.shape(), &[2, 3]);
    assert_eq!(c.data, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
}

#[test]
fn test_matmul_2d_zeros() {
    let a = Tensor::randn(&[3, 4]);
    let b = Tensor::zeros(&[4, 5]);
    let c = matmul(&a, &b);

    assert_eq!(c.shape(), &[3, 5]);
    assert!(c.data.iter().all(|&v| v == 0.0));
}

#[test]
fn test_matmul_2d_vector() {
    // [1, 3] @ [3, 1] → [1, 1] (dot product)
    let a = Tensor::new(vec![1.0, 2.0, 3.0], vec![1, 3]);
    let b = Tensor::new(vec![4.0, 5.0, 6.0], vec![3, 1]);
    let c = matmul(&a, &b);

    assert_eq!(c.shape(), &[1, 1]);
    assert_eq!(c.data, vec![32.0]); // 1*4 + 2*5 + 3*6
}

#[test]
fn test_matmul_with_transposed_input() {
    // A @ B.T where B is [4, 3], B.T is [3, 4]
    let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    let b = Tensor::new(
        vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
        vec![4, 3],
    );
    let bt = b.transpose(); // [3, 4]
    let c = matmul(&a, &bt);

    assert_eq!(c.shape(), &[2, 4]);
    // row 0: [1*1+2*0+3*0, 1*0+2*1+3*0, 1*0+2*0+3*1, 1*1+2*1+3*1] = [1, 2, 3, 6]
    // row 1: [4*1+5*0+6*0, 4*0+5*1+6*0, 4*0+5*0+6*1, 4*1+5*1+6*1] = [4, 5, 6, 15]
    assert_eq!(c.data, vec![1.0, 2.0, 3.0, 6.0, 4.0, 5.0, 6.0, 15.0]);
}

#[test]
#[should_panic(expected = "inner dimensions must match")]
fn test_matmul_2d_dimension_mismatch() {
    let a = Tensor::randn(&[2, 3]);
    let b = Tensor::randn(&[4, 5]);
    matmul(&a, &b);
}

#[test]
#[should_panic(expected = "2D or 3D")]
fn test_matmul_1d_panics() {
    let a = Tensor::randn(&[3]);
    let b = Tensor::randn(&[3]);
    matmul(&a, &b);
}

// ============================================================
// Cache-blocked matmul
// ============================================================

#[test]
fn test_matmul_blocked_matches_naive() {
    let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    let b = Tensor::new(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
        vec![3, 4],
    );
    let naive = matmul(&a, &b);
    let blocked = matmul_blocked(&a, &b);

    assert_eq!(naive.shape(), blocked.shape());
    assert_approx_eq(&naive.data, &blocked.data, 1e-6);
}

#[test]
fn test_matmul_blocked_larger_than_block_size() {
    // Matrix larger than block size (32) to exercise tiling
    let a = Tensor::randn(&[50, 40]);
    let b = Tensor::randn(&[40, 60]);

    let naive = matmul(&a, &b);
    let blocked = matmul_blocked(&a, &b);

    assert_eq!(naive.shape(), blocked.shape());
    assert_approx_eq(&naive.data, &blocked.data, 1e-4);
}

#[test]
fn test_matmul_blocked_non_divisible_by_block() {
    // Dimensions not evenly divisible by block size
    let a = Tensor::randn(&[37, 43]);
    let b = Tensor::randn(&[43, 29]);

    let naive = matmul(&a, &b);
    let blocked = matmul_blocked(&a, &b);

    assert_eq!(naive.shape(), blocked.shape());
    assert_approx_eq(&naive.data, &blocked.data, 1e-4);
}

#[test]
fn test_matmul_blocked_small() {
    // Smaller than one block
    let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    let b = Tensor::new(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]);
    let c = matmul_blocked(&a, &b);

    assert_eq!(c.shape(), &[2, 2]);
    assert_eq!(c.data, vec![19.0, 22.0, 43.0, 50.0]);
}

// ============================================================
// Batched matmul
// ============================================================

#[test]
fn test_batched_matmul_basic() {
    // batch=2, [2,2] @ [2,2] per batch
    let a = Tensor::new(
        vec![
            // batch 0
            1.0, 2.0, 3.0, 4.0,
            // batch 1
            5.0, 6.0, 7.0, 8.0,
        ],
        vec![2, 2, 2],
    );
    let b = Tensor::new(
        vec![
            // batch 0
            1.0, 0.0, 0.0, 1.0,
            // batch 1
            2.0, 0.0, 0.0, 2.0,
        ],
        vec![2, 2, 2],
    );
    let c = matmul(&a, &b);

    assert_eq!(c.shape(), &[2, 2, 2]);
    // batch 0: [[1,2],[3,4]] @ I = [[1,2],[3,4]]
    assert_eq!(c.get(&[0, 0, 0]), 1.0);
    assert_eq!(c.get(&[0, 0, 1]), 2.0);
    assert_eq!(c.get(&[0, 1, 0]), 3.0);
    assert_eq!(c.get(&[0, 1, 1]), 4.0);
    // batch 1: [[5,6],[7,8]] @ [[2,0],[0,2]] = [[10,12],[14,16]]
    assert_eq!(c.get(&[1, 0, 0]), 10.0);
    assert_eq!(c.get(&[1, 0, 1]), 12.0);
    assert_eq!(c.get(&[1, 1, 0]), 14.0);
    assert_eq!(c.get(&[1, 1, 1]), 16.0);
}

#[test]
fn test_batched_matmul_non_square() {
    // batch=1, [2, 3] @ [3, 4] → [1, 2, 4]
    let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 2, 3]);
    let b = Tensor::new(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
        vec![1, 3, 4],
    );
    let c = matmul(&a, &b);

    assert_eq!(c.shape(), &[1, 2, 4]);
    assert_eq!(c.data, vec![38.0, 44.0, 50.0, 56.0, 83.0, 98.0, 113.0, 128.0]);
}

#[test]
fn test_batched_matmul_matches_individual() {
    // Verify batched result matches doing each batch separately
    let a = Tensor::randn(&[4, 3, 5]);
    let b = Tensor::randn(&[4, 5, 2]);
    let c = batched_matmul(&a, &b);

    assert_eq!(c.shape(), &[4, 3, 2]);

    // Check each batch against individual 2D matmul
    for bi in 0..4 {
        for i in 0..3 {
            for j in 0..2 {
                let mut expected = 0.0;
                for p in 0..5 {
                    expected += a.get(&[bi, i, p]) * b.get(&[bi, p, j]);
                }
                assert!(
                    (c.get(&[bi, i, j]) - expected).abs() < 1e-4,
                    "mismatch at [{}, {}, {}]: {} vs {}",
                    bi, i, j, c.get(&[bi, i, j]), expected
                );
            }
        }
    }
}

#[test]
#[should_panic(expected = "batch dimensions must match")]
fn test_batched_matmul_batch_mismatch() {
    let a = Tensor::randn(&[2, 3, 4]);
    let b = Tensor::randn(&[3, 4, 5]);
    batched_matmul(&a, &b);
}

#[test]
#[should_panic(expected = "inner dimensions must match")]
fn test_batched_matmul_inner_mismatch() {
    let a = Tensor::randn(&[2, 3, 4]);
    let b = Tensor::randn(&[2, 5, 6]);
    batched_matmul(&a, &b);
}

// ============================================================
// Tensor method
// ============================================================

#[test]
fn test_tensor_matmul_method() {
    let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    let b = Tensor::new(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]);

    let c1 = matmul(&a, &b);
    let c2 = a.matmul(&b);
    assert_eq!(c1.data, c2.data);
}

// ============================================================
// Properties
// ============================================================

#[test]
fn test_matmul_associative() {
    // (A @ B) @ C == A @ (B @ C)
    let a = Tensor::randn(&[3, 4]);
    let b = Tensor::randn(&[4, 5]);
    let c = Tensor::randn(&[5, 2]);

    let ab_c = matmul(&matmul(&a, &b), &c);
    let a_bc = matmul(&a, &matmul(&b, &c));

    assert_eq!(ab_c.shape(), a_bc.shape());
    assert_approx_eq(&ab_c.data, &a_bc.data, 1e-3);
}

#[test]
fn test_matmul_distributive() {
    // A @ (B + C) == A @ B + A @ C
    let a = Tensor::randn(&[3, 4]);
    let b = Tensor::randn(&[4, 5]);
    let c = Tensor::randn(&[4, 5]);

    use bibe::tensor::ops;
    let bc_sum = ops::add(&b, &c);
    let lhs = matmul(&a, &bc_sum);
    let rhs = ops::add(&matmul(&a, &b), &matmul(&a, &c));

    assert_approx_eq(&lhs.data, &rhs.data, 1e-3);
}
