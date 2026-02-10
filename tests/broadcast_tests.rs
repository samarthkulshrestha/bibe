use bibe::tensor::Tensor;
use bibe::tensor::broadcast::{broadcast_shapes, broadcast_to, reduce_sum, reduce_mean, reduce_max};
use bibe::tensor::ops;

// ============================================================
// broadcast_shapes
// ============================================================

#[test]
fn test_broadcast_shapes_same() {
    assert_eq!(broadcast_shapes(&[3, 4], &[3, 4]), vec![3, 4]);
}

#[test]
fn test_broadcast_shapes_one_dim_is_1() {
    // [3, 1] + [1, 4] → [3, 4]
    assert_eq!(broadcast_shapes(&[3, 1], &[1, 4]), vec![3, 4]);
}

#[test]
fn test_broadcast_shapes_different_ndim() {
    // [2, 3, 4] + [4] → [2, 3, 4]
    assert_eq!(broadcast_shapes(&[2, 3, 4], &[4]), vec![2, 3, 4]);
}

#[test]
fn test_broadcast_shapes_both_expand() {
    // [2, 1, 4] + [3, 1] → [2, 3, 4]
    assert_eq!(broadcast_shapes(&[2, 1, 4], &[3, 1]), vec![2, 3, 4]);
}

#[test]
fn test_broadcast_shapes_scalar_like() {
    // [1] + [5] → [5]
    assert_eq!(broadcast_shapes(&[1], &[5]), vec![5]);
}

#[test]
fn test_broadcast_shapes_scalar_to_3d() {
    // [1] + [2, 3, 4] → [2, 3, 4]
    assert_eq!(broadcast_shapes(&[1], &[2, 3, 4]), vec![2, 3, 4]);
}

#[test]
#[should_panic(expected = "not broadcast-compatible")]
fn test_broadcast_shapes_incompatible() {
    broadcast_shapes(&[3, 4], &[3, 5]);
}

#[test]
#[should_panic(expected = "not broadcast-compatible")]
fn test_broadcast_shapes_incompatible_inner() {
    broadcast_shapes(&[2, 3], &[4, 3]);
}

// ============================================================
// broadcast_to
// ============================================================

#[test]
fn test_broadcast_to_noop() {
    let tensor = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]);
    let result = broadcast_to(&tensor, &[3]);
    assert_eq!(result.data, vec![1.0, 2.0, 3.0]);
}

#[test]
fn test_broadcast_to_row_vector() {
    // [1, 3] → [4, 3] (repeat rows)
    let tensor = Tensor::new(vec![1.0, 2.0, 3.0], vec![1, 3]);
    let result = broadcast_to(&tensor, &[4, 3]);
    assert_eq!(result.shape(), &[4, 3]);
    assert_eq!(result.data, vec![
        1.0, 2.0, 3.0,
        1.0, 2.0, 3.0,
        1.0, 2.0, 3.0,
        1.0, 2.0, 3.0,
    ]);
}

#[test]
fn test_broadcast_to_col_vector() {
    // [3, 1] → [3, 4] (repeat columns)
    let tensor = Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]);
    let result = broadcast_to(&tensor, &[3, 4]);
    assert_eq!(result.shape(), &[3, 4]);
    assert_eq!(result.data, vec![
        1.0, 1.0, 1.0, 1.0,
        2.0, 2.0, 2.0, 2.0,
        3.0, 3.0, 3.0, 3.0,
    ]);
}

#[test]
fn test_broadcast_to_add_leading_dim() {
    // [3] → [2, 3] (add batch dimension)
    let tensor = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]);
    let result = broadcast_to(&tensor, &[2, 3]);
    assert_eq!(result.shape(), &[2, 3]);
    assert_eq!(result.data, vec![
        1.0, 2.0, 3.0,
        1.0, 2.0, 3.0,
    ]);
}

#[test]
fn test_broadcast_to_scalar_to_matrix() {
    // [1, 1] → [2, 3]
    let tensor = Tensor::new(vec![5.0], vec![1, 1]);
    let result = broadcast_to(&tensor, &[2, 3]);
    assert_eq!(result.shape(), &[2, 3]);
    assert!(result.data.iter().all(|&v| v == 5.0));
}

// ============================================================
// Broadcasting through element-wise ops
// ============================================================

#[test]
fn test_add_broadcast_col_plus_row() {
    // [3, 1] + [1, 4] → [3, 4]
    let a = Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]);
    let b = Tensor::new(vec![10.0, 20.0, 30.0, 40.0], vec![1, 4]);
    let c = ops::add(&a, &b);

    assert_eq!(c.shape(), &[3, 4]);
    assert_eq!(c.data, vec![
        11.0, 21.0, 31.0, 41.0,
        12.0, 22.0, 32.0, 42.0,
        13.0, 23.0, 33.0, 43.0,
    ]);
}

#[test]
fn test_sub_broadcast() {
    // [2, 3] - [3] → [2, 3]
    let a = Tensor::new(vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0], vec![2, 3]);
    let b = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]);
    let c = ops::sub(&a, &b);

    assert_eq!(c.shape(), &[2, 3]);
    assert_eq!(c.data, vec![9.0, 18.0, 27.0, 39.0, 48.0, 57.0]);
}

#[test]
fn test_mul_broadcast() {
    // [2, 1] * [1, 3] → [2, 3]
    let a = Tensor::new(vec![2.0, 3.0], vec![2, 1]);
    let b = Tensor::new(vec![10.0, 20.0, 30.0], vec![1, 3]);
    let c = ops::mul(&a, &b);

    assert_eq!(c.shape(), &[2, 3]);
    assert_eq!(c.data, vec![20.0, 40.0, 60.0, 30.0, 60.0, 90.0]);
}

#[test]
fn test_div_broadcast() {
    // [2, 3] / [1, 3] → [2, 3]
    let a = Tensor::new(vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0], vec![2, 3]);
    let b = Tensor::new(vec![10.0, 10.0, 10.0], vec![1, 3]);
    let c = ops::div(&a, &b);

    assert_eq!(c.shape(), &[2, 3]);
    assert_eq!(c.data, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
}

#[test]
fn test_add_broadcast_scalar_like() {
    // [2, 3] + [1] → [2, 3]
    let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    let b = Tensor::new(vec![100.0], vec![1]);
    let c = ops::add(&a, &b);

    assert_eq!(c.shape(), &[2, 3]);
    assert_eq!(c.data, vec![101.0, 102.0, 103.0, 104.0, 105.0, 106.0]);
}

#[test]
fn test_add_same_shape_still_works() {
    let a = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]);
    let b = Tensor::new(vec![4.0, 5.0, 6.0], vec![3]);
    let c = ops::add(&a, &b);
    assert_eq!(c.data, vec![5.0, 7.0, 9.0]);
}

#[test]
fn test_operator_overload_with_broadcast() {
    let a = Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]);
    let b = Tensor::new(vec![10.0, 20.0], vec![1, 2]);
    let c = &a + &b;

    assert_eq!(c.shape(), &[3, 2]);
    assert_eq!(c.data, vec![11.0, 21.0, 12.0, 22.0, 13.0, 23.0]);
}

// ============================================================
// reduce_sum
// ============================================================

#[test]
fn test_reduce_sum_dim0() {
    // [[1, 2, 3],
    //  [4, 5, 6]]
    // sum(dim=0) → [5, 7, 9]
    let tensor = Tensor::new(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        vec![2, 3],
    );
    let result = reduce_sum(&tensor, 0);
    assert_eq!(result.shape(), &[3]);
    assert_eq!(result.data, vec![5.0, 7.0, 9.0]);
}

#[test]
fn test_reduce_sum_dim1() {
    // [[1, 2, 3],
    //  [4, 5, 6]]
    // sum(dim=1) → [6, 15]
    let tensor = Tensor::new(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        vec![2, 3],
    );
    let result = reduce_sum(&tensor, 1);
    assert_eq!(result.shape(), &[2]);
    assert_eq!(result.data, vec![6.0, 15.0]);
}

#[test]
fn test_reduce_sum_3d() {
    // shape [2, 2, 3], sum along dim=1
    let tensor = Tensor::new(
        vec![
            1.0, 2.0, 3.0,   // [0,0,:]
            4.0, 5.0, 6.0,   // [0,1,:]
            7.0, 8.0, 9.0,   // [1,0,:]
            10.0, 11.0, 12.0, // [1,1,:]
        ],
        vec![2, 2, 3],
    );
    let result = reduce_sum(&tensor, 1);
    assert_eq!(result.shape(), &[2, 3]);
    // [0,:,:] sum → [5, 7, 9]
    // [1,:,:] sum → [17, 19, 21]
    assert_eq!(result.data, vec![5.0, 7.0, 9.0, 17.0, 19.0, 21.0]);
}

#[test]
fn test_reduce_sum_1d() {
    let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![4]);
    let result = reduce_sum(&tensor, 0);
    assert_eq!(result.shape(), &[1]);
    assert_eq!(result.data, vec![10.0]);
}

#[test]
#[should_panic(expected = "out of range")]
fn test_reduce_sum_invalid_dim() {
    let tensor = Tensor::zeros(&[2, 3]);
    reduce_sum(&tensor, 2);
}

// ============================================================
// reduce_mean
// ============================================================

#[test]
fn test_reduce_mean_dim0() {
    let tensor = Tensor::new(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        vec![2, 3],
    );
    let result = reduce_mean(&tensor, 0);
    assert_eq!(result.shape(), &[3]);
    assert_eq!(result.data, vec![2.5, 3.5, 4.5]);
}

#[test]
fn test_reduce_mean_dim1() {
    let tensor = Tensor::new(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        vec![2, 3],
    );
    let result = reduce_mean(&tensor, 1);
    assert_eq!(result.shape(), &[2]);
    assert_eq!(result.data, vec![2.0, 5.0]);
}

// ============================================================
// reduce_max
// ============================================================

#[test]
fn test_reduce_max_dim0() {
    let tensor = Tensor::new(
        vec![1.0, 5.0, 3.0, 4.0, 2.0, 6.0],
        vec![2, 3],
    );
    let result = reduce_max(&tensor, 0);
    assert_eq!(result.shape(), &[3]);
    assert_eq!(result.data, vec![4.0, 5.0, 6.0]);
}

#[test]
fn test_reduce_max_dim1() {
    let tensor = Tensor::new(
        vec![1.0, 5.0, 3.0, 4.0, 2.0, 6.0],
        vec![2, 3],
    );
    let result = reduce_max(&tensor, 1);
    assert_eq!(result.shape(), &[2]);
    assert_eq!(result.data, vec![5.0, 6.0]);
}

#[test]
fn test_reduce_max_with_negatives() {
    let tensor = Tensor::new(vec![-3.0, -1.0, -5.0, -2.0], vec![2, 2]);
    let result = reduce_max(&tensor, 1);
    assert_eq!(result.shape(), &[2]);
    assert_eq!(result.data, vec![-1.0, -2.0]);
}

#[test]
fn test_reduce_max_1d() {
    let tensor = Tensor::new(vec![3.0, 1.0, 4.0, 1.0, 5.0], vec![5]);
    let result = reduce_max(&tensor, 0);
    assert_eq!(result.shape(), &[1]);
    assert_eq!(result.data, vec![5.0]);
}

// ============================================================
// Combined: broadcast + reduce roundtrip
// ============================================================

#[test]
fn test_broadcast_then_reduce_recovers_original() {
    // Broadcasting [3] → [4, 3] then reducing dim=0 should give 4 * original
    let tensor = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]);
    let broadcasted = broadcast_to(&tensor, &[4, 3]);
    let reduced = reduce_sum(&broadcasted, 0);

    assert_eq!(reduced.shape(), &[3]);
    assert_eq!(reduced.data, vec![4.0, 8.0, 12.0]);
}

#[test]
fn test_reduce_sum_all_dims() {
    // Sum a 2x3 matrix completely by reducing both dims
    let tensor = Tensor::new(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        vec![2, 3],
    );
    let sum_dim1 = reduce_sum(&tensor, 1);  // [6.0, 15.0]
    let sum_all = reduce_sum(&sum_dim1, 0); // [21.0]
    assert_eq!(sum_all.data, vec![21.0]);
}
