use crate::tensor::Tensor;
use crate::tensor::broadcast::{reduce_sum, reduce_max};
use crate::tensor::ops;

/// Insert a size-1 dimension at `dim` after a reduction.
fn unsqueeze(tensor: &Tensor, dim: usize) -> Tensor {
    let mut new_shape = tensor.shape().to_vec();
    new_shape.insert(dim, 1);
    tensor.reshape(&new_shape)
}

fn reduce_max_keepdim(tensor: &Tensor, dim: usize) -> Tensor {
    let reduced = reduce_max(tensor, dim);
    unsqueeze(&reduced, dim)
}

fn reduce_sum_keepdim(tensor: &Tensor, dim: usize) -> Tensor {
    let reduced = reduce_sum(tensor, dim);
    unsqueeze(&reduced, dim)
}

/// Log-sum-exp along a dimension (numerically stable).
///   logsumexp(x, dim) = max + log(sum(exp(x - max), dim))
/// where max = max(x, dim, keepdim=true)
pub fn logsumexp(tensor: &Tensor, dim: usize) -> Tensor {
    let max_val = reduce_max_keepdim(tensor, dim);
    let shifted = ops::sub(tensor, &max_val);
    let exp_shifted = ops::exp(&shifted);
    let sum_exp = reduce_sum(&exp_shifted, dim);
    let log_sum = ops::log(&sum_exp);

    // max_val was keepdim, squeeze it back to match log_sum shape
    let max_squeezed = reduce_max(tensor, dim);
    ops::add(&max_squeezed, &log_sum)
}

/// Numerically stable softmax along a dimension.
///   softmax(x, dim) = exp(x - max) / sum(exp(x - max), dim, keepdim=true)
pub fn stable_softmax(tensor: &Tensor, dim: usize) -> Tensor {
    let max_val = reduce_max_keepdim(tensor, dim);
    let shifted = ops::sub(tensor, &max_val);
    let exp_shifted = ops::exp(&shifted);
    let sum_exp = reduce_sum_keepdim(&exp_shifted, dim);
    ops::div(&exp_shifted, &sum_exp)
}

pub fn clip(tensor: &Tensor, min: f32, max: f32) -> Tensor {
    let data: Vec<f32> = tensor.data.iter()
        .map(|&x| x.clamp(min, max))
        .collect();
    Tensor::new(data, tensor.shape().to_vec())
}

/// Logarithm with a safety floor: ln(max(x, epsilon)).
pub fn safe_log(tensor: &Tensor, epsilon: f32) -> Tensor {
    let data: Vec<f32> = tensor.data.iter()
        .map(|&x| x.max(epsilon).ln())
        .collect();
    Tensor::new(data, tensor.shape().to_vec())
}

pub fn has_nan(tensor: &Tensor) -> bool {
    tensor.data.iter().any(|x| x.is_nan())
}

pub fn has_inf(tensor: &Tensor) -> bool {
    tensor.data.iter().any(|x| x.is_infinite())
}

pub fn all_finite(tensor: &Tensor) -> bool {
    tensor.data.iter().all(|x| x.is_finite())
}

impl Tensor {
    pub fn softmax(&self, dim: usize) -> Tensor {
        stable_softmax(self, dim)
    }

    pub fn logsumexp(&self, dim: usize) -> Tensor {
        logsumexp(self, dim)
    }

    pub fn clip(&self, min: f32, max: f32) -> Tensor {
        clip(self, min, max)
    }
}
