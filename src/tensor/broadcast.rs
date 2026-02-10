use crate::tensor::Tensor;

/// Compute the broadcast-compatible output shape from two input shapes.
/// Returns None if shapes are not broadcast-compatible
pub fn broadcast_shapes(shape_a: &[usize], shape_b: &[usize]) -> Vec<usize> {
    let max_ndim = shape_a.len().max(shape_b.len());
    let mut result = vec![0; max_ndim];

    // Walk from right, treating missing dims as 1
    for i in 0..max_ndim {
        let da = if i < shape_a.len() {
            shape_a[shape_a.len() - 1 - i]
        } else {
            1
        };
        let db = if i < shape_b.len() {
            shape_b[shape_b.len() - 1 - i]
        } else {
            1
        };

        assert!(
            da == db || da == 1 || db == 1,
            "shapes {:?} and {:?} are not broadcast-compatible at dimension {}.",
            shape_a, shape_b, max_ndim - 1 - i
        );

        result[max_ndim - 1 - i] = da.max(db);
    }
    result
}

/// Expand a tensor to a target shape by repeating along size-1 dimensions.
/// The target shape must be broadcast-compatible with the tensor's shape.
pub fn broadcast_to(tensor: &Tensor, target_shape: &[usize]) -> Tensor {
    if tensor.shape() == target_shape {
        return tensor.clone();
    }

    let src_shape = tensor.shape();
    let ndim = target_shape.len();

    let mut padded_shape = vec![1; ndim];
    let offset = ndim - src_shape.len();
    padded_shape[offset..].copy_from_slice(src_shape);

    let target_size: usize = target_shape.iter().product();
    let mut data = Vec::with_capacity(target_size);

    let mut target_indices = vec![0usize; ndim];

    let src_strides = compute_strides_for(&padded_shape);

    for _ in 0..target_size {
        let mut src_flat = 0;
        for d in 0..ndim {
            let src_idx = if padded_shape[d] == 1 { 0 } else { target_indices[d] };
            src_flat += src_idx * src_strides[d];
        }

        data.push(tensor.data[src_flat]);

        for d in (0..ndim).rev() {
            target_indices[d] += 1;
            if target_indices[d] < target_shape[d] {
                break;
            }
            target_indices[d] = 0;
        }
    }

    Tensor::new(data, target_shape.to_vec())
}

fn compute_strides_for(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![1; shape.len()];
    for i in (0..shape.len().saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}

/// Sum along a dimension, removing that dimension from the shape.
/// Example: [2, 3, 4] reduce_sum dim = 1 → [2, 4]
pub fn reduce_sum(tensor: &Tensor, dim: usize) -> Tensor {
    let shape = tensor.shape();
    assert!(dim < shape.len(), "dim {dim} out of range for {}D tensor", shape.len());

    let mut out_shape: Vec<usize> = shape.to_vec();
    out_shape.remove(dim);
    if out_shape.is_empty() {
        out_shape.push(1); // scalar result
    }

    let out_size: usize = out_shape.iter().product();
    let mut data = vec![0.0; out_size];

    let ndim = shape.len();
    let mut indices = vec![0usize; ndim];
    let total: usize = shape.iter().product();
    let out_strides = compute_strides_for(&out_shape);

    for _ in 0..total {
        // Compute output flat index (skip the reduced dimension)
        let out_indices: Vec<usize> = indices.iter().enumerate()
            .filter(|&(d, _)| d != dim)
            .map(|(_, &idx)| idx)
            .collect();

        let out_flat: usize = out_indices.iter()
            .zip(out_strides.iter())
            .map(|(&idx, &stride)| idx * stride)
            .sum();

        data[out_flat] += tensor.get(&indices);

        for d in (0..ndim).rev() {
            indices[d] += 1;
            if indices[d] < shape[d] {
                break;
            }
            indices[d] = 0;
        }
    }

    Tensor::new(data, out_shape)
}

/// Mean along a dimension, removing that dimension from the shape.
pub fn reduce_mean(tensor: &Tensor, dim: usize) -> Tensor {
    let n = tensor.shape()[dim] as f32;
    let sum = reduce_sum(tensor, dim);
    crate::tensor::ops::div_scalar(&sum, n)
}

/// Max along a dimension, removing that dimension from the shape.
pub fn reduce_max(tensor: &Tensor, dim: usize) -> Tensor {
    let shape = tensor.shape();
    assert!(dim < shape.len(), "dim {dim} out of range for {}D tensor", shape.len());

    let mut out_shape: Vec<usize> = shape.to_vec();
    out_shape.remove(dim);
    if out_shape.is_empty() {
        out_shape.push(1);
    }

    let out_size: usize = out_shape.iter().product();
    let mut data = vec![f32::NEG_INFINITY; out_size];

    let ndim = shape.len();
    let mut indices = vec![0usize; ndim];
    let total: usize = shape.iter().product();
    let out_strides = compute_strides_for(&out_shape);

    for _ in 0..total {
        let out_indices: Vec<usize> = indices.iter().enumerate()
            .filter(|&(d, _)| d != dim)
            .map(|(_, &idx)| idx)
            .collect();

        let out_flat: usize = out_indices.iter()
            .zip(out_strides.iter())
            .map(|(&idx, &stride)| idx * stride)
            .sum();

        data[out_flat] = data[out_flat].max(tensor.get(&indices));

        for d in (0..ndim).rev() {
            indices[d] += 1;
            if indices[d] < shape[d] {
                break;
            }
            indices[d] = 0;
        }
    }

    Tensor::new(data, out_shape)
}
