use crate::tensor::Tensor;
use crate::tensor::broadcast::{broadcast_to, reduce_sum};
use crate::tensor::matmul::matmul;
use crate::tensor::ops;
use super::node::GradFn;

// ============================================================
// Helper: reduce a gradient to match a (possibly broadcast) shape
// ============================================================

/// When `a` was broadcast from `target_shape` to `grad.shape()` during
/// the forward pass, the backward pass must sum out the broadcast dims
/// to get a gradient of the original shape.
pub(crate) fn reduce_to_shape(grad: &Tensor, target_shape: &[usize]) -> Tensor {
    let grad_shape = grad.shape();
    if grad_shape == target_shape {
        return grad.clone();
    }

    let ndim = grad_shape.len();
    let mut padded = vec![1usize; ndim];
    let offset = ndim.saturating_sub(target_shape.len());
    padded[offset..].copy_from_slice(target_shape);

    let mut result = grad.clone();
    for d in (0..ndim).rev() {
        if padded[d] == 1 && result.shape()[d] > 1 {
            result = reduce_sum(&result, d);
        }
    }

    if result.shape() != target_shape {
        result = result.reshape(target_shape);
    }

    result
}

// ============================================================
// Backward function structs
// ============================================================

// --- Add: d(a+b)/da = 1, d(a+b)/db = 1 ---

pub(crate) struct AddBackward {
    pub a_shape: Vec<usize>,
    pub b_shape: Vec<usize>,
}

impl GradFn for AddBackward {
    fn backward(&self, grad_output: &Tensor) -> Vec<Tensor> {
        let grad_a = reduce_to_shape(grad_output, &self.a_shape);
        let grad_b = reduce_to_shape(grad_output, &self.b_shape);
        vec![grad_a, grad_b]
    }
}

// --- Sub: d(a-b)/da = 1, d(a-b)/db = -1 ---

pub(crate) struct SubBackward {
    pub a_shape: Vec<usize>,
    pub b_shape: Vec<usize>,
}

impl GradFn for SubBackward {
    fn backward(&self, grad_output: &Tensor) -> Vec<Tensor> {
        let grad_a = reduce_to_shape(grad_output, &self.a_shape);
        let grad_b = reduce_to_shape(&ops::neg(grad_output), &self.b_shape);
        vec![grad_a, grad_b]
    }
}

// --- Mul (element-wise): d(a*b)/da = b, d(a*b)/db = a ---

pub(crate) struct MulBackward {
    pub a: Tensor,
    pub b: Tensor,
    pub a_shape: Vec<usize>,
    pub b_shape: Vec<usize>,
}

impl GradFn for MulBackward {
    fn backward(&self, grad_output: &Tensor) -> Vec<Tensor> {
        let grad_a = reduce_to_shape(&ops::mul(grad_output, &self.b), &self.a_shape);
        let grad_b = reduce_to_shape(&ops::mul(grad_output, &self.a), &self.b_shape);
        vec![grad_a, grad_b]
    }
}

// --- Div (element-wise): d(a/b)/da = 1/b, d(a/b)/db = -a/b² ---

pub(crate) struct DivBackward {
    pub a: Tensor,
    pub b: Tensor,
    pub a_shape: Vec<usize>,
    pub b_shape: Vec<usize>,
}

impl GradFn for DivBackward {
    fn backward(&self, grad_output: &Tensor) -> Vec<Tensor> {
        // grad_a = grad_out / b
        let grad_a = reduce_to_shape(&ops::div(grad_output, &self.b), &self.a_shape);
        // grad_b = -grad_out * a / b²
        let neg_grad = ops::neg(grad_output);
        let grad_b = reduce_to_shape(
            &ops::div(&ops::mul(&neg_grad, &self.a), &ops::mul(&self.b, &self.b)),
            &self.b_shape,
        );
        vec![grad_a, grad_b]
    }
}

// --- Neg: d(-a)/da = -1 ---

pub(crate) struct NegBackward;

impl GradFn for NegBackward {
    fn backward(&self, grad_output: &Tensor) -> Vec<Tensor> {
        vec![ops::neg(grad_output)]
    }
}

// --- Exp: d(exp(x))/dx = exp(x) ---

pub(crate) struct ExpBackward {
    pub output: Tensor,
}

impl GradFn for ExpBackward {
    fn backward(&self, grad_output: &Tensor) -> Vec<Tensor> {
        vec![ops::mul(grad_output, &self.output)]
    }
}

// --- Log: d(ln(x))/dx = 1/x ---

pub(crate) struct LogBackward {
    pub input: Tensor,
}

impl GradFn for LogBackward {
    fn backward(&self, grad_output: &Tensor) -> Vec<Tensor> {
        vec![ops::div(grad_output, &self.input)]
    }
}

// --- Sqrt: d(√x)/dx = 1/(2√x) ---

pub(crate) struct SqrtBackward {
    pub output: Tensor,
}

impl GradFn for SqrtBackward {
    fn backward(&self, grad_output: &Tensor) -> Vec<Tensor> {
        // grad = grad_out / (2 * sqrt(x))  =  grad_out / (2 * output)
        let two_output = ops::mul_scalar(&self.output, 2.0);
        vec![ops::div(grad_output, &two_output)]
    }
}

// --- Pow: d(x^p)/dx = p * x^(p-1) ---

pub(crate) struct PowBackward {
    pub input: Tensor,
    pub power: f32,
}

impl GradFn for PowBackward {
    fn backward(&self, grad_output: &Tensor) -> Vec<Tensor> {
        let inner = ops::mul_scalar(&ops::pow(&self.input, self.power - 1.0), self.power);
        vec![ops::mul(grad_output, &inner)]
    }
}

// --- Sum (reduce to scalar): grad = broadcast(grad_out, original_shape) ---

pub(crate) struct SumBackward {
    pub input_shape: Vec<usize>,
}

impl GradFn for SumBackward {
    fn backward(&self, grad_output: &Tensor) -> Vec<Tensor> {
        vec![broadcast_to(grad_output, &self.input_shape)]
    }
}

// --- Matmul: dL/dA = dL/dC @ B^T, dL/dB = A^T @ dL/dC ---

pub(crate) struct MatmulBackward {
    pub a: Tensor,
    pub b: Tensor,
}

impl GradFn for MatmulBackward {
    fn backward(&self, grad_output: &Tensor) -> Vec<Tensor> {
        let grad_a = matmul(grad_output, &self.b.transpose_contiguous());
        let grad_b = matmul(&self.a.transpose_contiguous(), grad_output);
        vec![grad_a, grad_b]
    }
}

// --- MulScalar: d(a*c)/da = c ---

pub(crate) struct MulScalarBackward {
    pub scalar: f32,
}

impl GradFn for MulScalarBackward {
    fn backward(&self, grad_output: &Tensor) -> Vec<Tensor> {
        vec![ops::mul_scalar(grad_output, self.scalar)]
    }
}

// --- Transpose: d(A^T) = (dL/dA^T)^T ---

pub(crate) struct TransposeBackward;

impl GradFn for TransposeBackward {
    fn backward(&self, grad_output: &Tensor) -> Vec<Tensor> {
        vec![grad_output.transpose_contiguous()]
    }
}

// --- Softmax: grad = s * (grad_out - sum(grad_out * s, dim, keepdim)) ---

pub(crate) struct SoftmaxBackward {
    pub output: Tensor, // softmax(x)
    pub dim: usize,
}

impl GradFn for SoftmaxBackward {
    fn backward(&self, grad_output: &Tensor) -> Vec<Tensor> {
        // s = softmax output
        // ds_i/dx_j = s_i * (δ_ij - s_j)
        // dL/dx = s * (dL/ds - sum(dL/ds * s, dim, keepdim))
        let gs = ops::mul(grad_output, &self.output);

        // sum(grad_out * s) along softmax dim, keepdim
        let mut sum_shape = gs.shape().to_vec();
        let sum_reduced = reduce_sum(&gs, self.dim);
        sum_shape[self.dim] = 1;
        let sum_keepdim = sum_reduced.reshape(&sum_shape);

        let shifted = ops::sub(grad_output, &sum_keepdim);
        vec![ops::mul(&self.output, &shifted)]
    }
}
