use bibe::tensor::Tensor;

// ============================================================
// Tensor::new - constructor
// ============================================================

#[test]
fn test_new_creates_tensor_with_correct_data() {
    let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    assert_eq!(tensor.data, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
}

#[test]
fn test_new_1d_tensor() {
    let tensor = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]);
    assert_eq!(tensor.get(&[0]), 1.0);
    assert_eq!(tensor.get(&[2]), 3.0);
}

#[test]
fn test_new_3d_tensor() {
    let data: Vec<f32> = (1..=24).map(|x| x as f32).collect();
    let tensor = Tensor::new(data, vec![2, 3, 4]);
    assert_eq!(tensor.get(&[0, 0, 0]), 1.0);
    assert_eq!(tensor.get(&[1, 2, 3]), 24.0);
}

#[test]
#[should_panic]
fn test_new_mismatched_data_and_shape() {
    Tensor::new(vec![1.0, 2.0, 3.0], vec![2, 3]);
}

// ============================================================
// Tensor::zeros and Tensor::ones
// ============================================================

#[test]
fn test_zeros_1d() {
    let tensor = Tensor::zeros(&[5]);
    assert_eq!(tensor.data, vec![0.0; 5]);
}

#[test]
fn test_zeros_2d() {
    let tensor = Tensor::zeros(&[3, 4]);
    assert_eq!(tensor.data.len(), 12);
    assert!(tensor.data.iter().all(|&v| v == 0.0));
}

#[test]
fn test_zeros_3d() {
    let tensor = Tensor::zeros(&[2, 3, 4]);
    assert_eq!(tensor.data.len(), 24);
    assert!(tensor.data.iter().all(|&v| v == 0.0));
}

#[test]
fn test_ones_1d() {
    let tensor = Tensor::ones(&[5]);
    assert_eq!(tensor.data, vec![1.0; 5]);
}

#[test]
fn test_ones_2d() {
    let tensor = Tensor::ones(&[2, 3]);
    assert_eq!(tensor.data.len(), 6);
    assert!(tensor.data.iter().all(|&v| v == 1.0));
}

// ============================================================
// Tensor::randn
// ============================================================

#[test]
fn test_randn_correct_shape() {
    let tensor = Tensor::randn(&[3, 4]);
    assert_eq!(tensor.data.len(), 12);
}

#[test]
fn test_randn_not_all_zeros() {
    let tensor = Tensor::randn(&[100]);
    assert!(tensor.data.iter().any(|&v| v != 0.0));
}

#[test]
fn test_randn_approximately_standard_normal() {
    let tensor = Tensor::randn(&[10000]);
    let mean: f32 = tensor.data.iter().sum::<f32>() / tensor.data.len() as f32;
    let variance: f32 = tensor.data.iter().map(|x| (x - mean).powi(2)).sum::<f32>()
        / tensor.data.len() as f32;

    assert!((mean).abs() < 0.1, "mean {mean} not close to 0");
    assert!((variance - 1.0).abs() < 0.1, "variance {variance} not close to 1");
}

// ============================================================
// Xavier initialization
// ============================================================

#[test]
fn test_xaviern_correct_shape() {
    let tensor = Tensor::xaviern(&[64, 128]);
    assert_eq!(tensor.data.len(), 64 * 128);
}

#[test]
fn test_xaviern_variance_scales_with_fan() {
    let tensor = Tensor::xaviern(&[512, 256]);
    let mean: f32 = tensor.data.iter().sum::<f32>() / tensor.data.len() as f32;
    let variance: f32 = tensor.data.iter().map(|x| (x - mean).powi(2)).sum::<f32>()
        / tensor.data.len() as f32;

    // Xavier normal: variance should be ~2/(fan_in + fan_out) = 2/768
    let expected_variance = 2.0 / (256.0 + 512.0);
    assert!(
        (variance - expected_variance).abs() < 0.001,
        "variance {variance} not close to expected {expected_variance}"
    );
}

#[test]
#[should_panic(expected = "xavier init requires at least 2D shape")]
fn test_xaviern_1d_panics() {
    Tensor::xaviern(&[10]);
}

#[test]
fn test_xavieru_within_bounds() {
    let tensor = Tensor::xavieru(&[64, 128]);
    let fan_in = 128.0_f32;
    let fan_out = 64.0_f32;
    let limit = (6.0 / (fan_in + fan_out)).sqrt();

    assert!(tensor.data.iter().all(|&v| v >= -limit && v <= limit));
}

// ============================================================
// Strides / shape calculation
// ============================================================

#[test]
fn test_strides_1d() {
    let tensor = Tensor::zeros(&[5]);
    // 1D tensor: stride is [1]
    assert_eq!(tensor.get(&[0]), 0.0);
    assert_eq!(tensor.get(&[4]), 0.0);
}

#[test]
fn test_strides_2d_row_major() {
    let tensor = Tensor::new(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        vec![2, 3],
    );
    // Row-major: strides should be [3, 1]
    // [0,0]=1, [0,1]=2, [0,2]=3, [1,0]=4, [1,1]=5, [1,2]=6
    assert_eq!(tensor.get(&[0, 0]), 1.0);
    assert_eq!(tensor.get(&[0, 1]), 2.0);
    assert_eq!(tensor.get(&[0, 2]), 3.0);
    assert_eq!(tensor.get(&[1, 0]), 4.0);
    assert_eq!(tensor.get(&[1, 1]), 5.0);
    assert_eq!(tensor.get(&[1, 2]), 6.0);
}

#[test]
fn test_strides_3d_row_major() {
    let tensor = Tensor::new(
        (1..=24).map(|x| x as f32).collect(),
        vec![2, 3, 4],
    );
    // Strides: [12, 4, 1]
    assert_eq!(tensor.get(&[0, 0, 0]), 1.0);
    assert_eq!(tensor.get(&[0, 0, 3]), 4.0);
    assert_eq!(tensor.get(&[0, 1, 0]), 5.0);
    assert_eq!(tensor.get(&[1, 0, 0]), 13.0);
    assert_eq!(tensor.get(&[1, 2, 3]), 24.0);
}

// ============================================================
// Indexing: get and set
// ============================================================

#[test]
fn test_get_set_2d() {
    let mut tensor = Tensor::zeros(&[3, 4]);
    tensor.set(&[1, 2], 42.0);
    assert_eq!(tensor.get(&[1, 2]), 42.0);
    assert_eq!(tensor.get(&[0, 0]), 0.0);
}

#[test]
fn test_set_all_elements() {
    let mut tensor = Tensor::zeros(&[2, 3]);
    let mut val = 1.0;
    for i in 0..2 {
        for j in 0..3 {
            tensor.set(&[i, j], val);
            val += 1.0;
        }
    }
    assert_eq!(tensor.get(&[0, 0]), 1.0);
    assert_eq!(tensor.get(&[1, 2]), 6.0);
}

#[test]
#[should_panic(expected = "out of bounds")]
fn test_get_out_of_bounds() {
    let tensor = Tensor::zeros(&[2, 3]);
    tensor.get(&[2, 0]);
}

#[test]
#[should_panic(expected = "don't match")]
fn test_get_wrong_dimensions() {
    let tensor = Tensor::zeros(&[2, 3]);
    tensor.get(&[0, 0, 0]);
}

#[test]
fn test_index_trait() {
    let mut tensor = Tensor::new(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        vec![2, 3],
    );
    assert_eq!(tensor[[0, 0]], 1.0);
    assert_eq!(tensor[[1, 2]], 6.0);

    tensor[[1, 1]] = 99.0;
    assert_eq!(tensor[[1, 1]], 99.0);
}

// ============================================================
// Transpose (2D)
// ============================================================

#[test]
fn test_transpose_shape() {
    let tensor = Tensor::randn(&[3, 5]);
    let t = tensor.transpose();
    assert_eq!(t.data.len(), 15);
    assert_eq!(t.get(&[0, 0]), tensor.get(&[0, 0]));
}

#[test]
fn test_transpose_values() {
    let tensor = Tensor::new(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        vec![2, 3],
    );
    // [[1, 2, 3], [4, 5, 6]] -> [[1, 4], [2, 5], [3, 6]]
    let t = tensor.transpose();
    assert_eq!(t.get(&[0, 0]), 1.0);
    assert_eq!(t.get(&[0, 1]), 4.0);
    assert_eq!(t.get(&[1, 0]), 2.0);
    assert_eq!(t.get(&[1, 1]), 5.0);
    assert_eq!(t.get(&[2, 0]), 3.0);
    assert_eq!(t.get(&[2, 1]), 6.0);
}

#[test]
fn test_double_transpose_identity() {
    let tensor = Tensor::new(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        vec![2, 3],
    );
    let tt = tensor.transpose().transpose();
    for i in 0..2 {
        for j in 0..3 {
            assert_eq!(tensor.get(&[i, j]), tt.get(&[i, j]));
        }
    }
}

#[test]
fn test_transpose_shorthand() {
    let tensor = Tensor::randn(&[4, 7]);
    let t1 = tensor.transpose();
    let t2 = tensor.t();
    for i in 0..7 {
        for j in 0..4 {
            assert_eq!(t1.get(&[i, j]), t2.get(&[i, j]));
        }
    }
}

#[test]
fn test_transpose_is_non_contiguous() {
    let tensor = Tensor::randn(&[3, 4]);
    let t = tensor.transpose();
    assert!(!t.is_contiguous());
}

#[test]
fn test_transpose_contiguous_values() {
    let tensor = Tensor::new(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        vec![2, 3],
    );
    let tc = tensor.transpose_contiguous();
    assert!(tc.is_contiguous());
    assert_eq!(tc.get(&[0, 0]), 1.0);
    assert_eq!(tc.get(&[0, 1]), 4.0);
    assert_eq!(tc.get(&[1, 0]), 2.0);
    assert_eq!(tc.get(&[2, 1]), 6.0);
}

#[test]
#[should_panic(expected = "2D")]
fn test_transpose_non_2d_panics() {
    let tensor = Tensor::randn(&[2, 3, 4]);
    tensor.transpose();
}

// ============================================================
// Reshape
// ============================================================

#[test]
fn test_reshape_2d_to_2d() {
    let tensor = Tensor::new(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        vec![2, 3],
    );
    let reshaped = tensor.reshape(&[3, 2]);
    assert_eq!(reshaped.get(&[0, 0]), 1.0);
    assert_eq!(reshaped.get(&[0, 1]), 2.0);
    assert_eq!(reshaped.get(&[1, 0]), 3.0);
    assert_eq!(reshaped.get(&[2, 1]), 6.0);
}

#[test]
fn test_reshape_flatten() {
    let tensor = Tensor::new(
        (1..=12).map(|x| x as f32).collect(),
        vec![3, 4],
    );
    let flat = tensor.reshape(&[12]);
    assert_eq!(flat.get(&[0]), 1.0);
    assert_eq!(flat.get(&[11]), 12.0);
}

#[test]
fn test_reshape_to_3d() {
    let tensor = Tensor::new(
        (1..=24).map(|x| x as f32).collect(),
        vec![24],
    );
    let reshaped = tensor.reshape(&[2, 3, 4]);
    assert_eq!(reshaped.get(&[0, 0, 0]), 1.0);
    assert_eq!(reshaped.get(&[1, 2, 3]), 24.0);
}

#[test]
fn test_reshape_preserves_element_order() {
    let data: Vec<f32> = (1..=12).map(|x| x as f32).collect();
    let tensor = Tensor::new(data.clone(), vec![3, 4]);
    let reshaped = tensor.reshape(&[2, 6]);
    assert_eq!(reshaped.data, data);
}

#[test]
#[should_panic(expected = "cannot reshape tensor of size")]
fn test_reshape_wrong_size() {
    let tensor = Tensor::zeros(&[2, 3]);
    tensor.reshape(&[2, 2]);
}

#[test]
#[should_panic(expected = "non-contiguous")]
fn test_reshape_non_contiguous_panics() {
    let tensor = Tensor::randn(&[3, 4]);
    let transposed = tensor.transpose();
    transposed.reshape(&[12]);
}

#[test]
fn test_reshape_non_contiguous_via_contiguous() {
    let tensor = Tensor::new(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        vec![2, 3],
    );
    let transposed = tensor.transpose();
    let reshaped = transposed.contiguous().reshape(&[6]);

    // Transpose: [[1,4],[2,5],[3,6]], flattened: [1,4,2,5,3,6]
    assert_eq!(reshaped.get(&[0]), 1.0);
    assert_eq!(reshaped.get(&[1]), 4.0);
    assert_eq!(reshaped.get(&[2]), 2.0);
    assert_eq!(reshaped.get(&[3]), 5.0);
    assert_eq!(reshaped.get(&[4]), 3.0);
    assert_eq!(reshaped.get(&[5]), 6.0);
}

// ============================================================
// Contiguity
// ============================================================

#[test]
fn test_new_tensor_is_contiguous() {
    let tensor = Tensor::randn(&[3, 4, 5]);
    assert!(tensor.is_contiguous());
}

#[test]
fn test_contiguous_copy_preserves_values() {
    let tensor = Tensor::new(
        vec![1.0, 2.0, 3.0, 4.0],
        vec![2, 2],
    );
    let t = tensor.transpose();
    let c = t.contiguous();

    assert!(c.is_contiguous());
    for i in 0..2 {
        for j in 0..2 {
            assert_eq!(t.get(&[i, j]), c.get(&[i, j]));
        }
    }
}

#[test]
fn test_contiguous_on_contiguous_is_noop() {
    let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    let c = tensor.contiguous();
    assert_eq!(tensor.data, c.data);
}

// ============================================================
// Clone
// ============================================================

#[test]
fn test_clone_independence() {
    let tensor = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]);
    let mut cloned = tensor.clone();
    cloned.set(&[0], 99.0);
    assert_eq!(tensor.get(&[0]), 1.0);
    assert_eq!(cloned.get(&[0]), 99.0);
}

// ============================================================
// He initialization
// ============================================================

#[test]
fn test_hen_correct_shape() {
    let tensor = Tensor::hen(&[64, 128]);
    assert_eq!(tensor.data.len(), 64 * 128);
}

#[test]
fn test_hen_variance_scales_with_fan_in() {
    let tensor = Tensor::hen(&[512, 256]);
    let mean: f32 = tensor.data.iter().sum::<f32>() / tensor.data.len() as f32;
    let variance: f32 = tensor.data.iter().map(|x| (x - mean).powi(2)).sum::<f32>()
        / tensor.data.len() as f32;

    // He normal: variance should be ~2/fan_in = 2/256
    let expected_variance = 2.0 / 256.0;
    assert!(
        (variance - expected_variance).abs() < 0.002,
        "variance {variance} not close to expected {expected_variance}"
    );
}

#[test]
#[should_panic(expected = "he init requires at least 2D shape")]
fn test_hen_1d_panics() {
    Tensor::hen(&[10]);
}
