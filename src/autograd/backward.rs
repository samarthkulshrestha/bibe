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

// --- MeanDim: mean along `dim`, keeping that dim as size 1 ---
// d(mean_k)/d(x_i) = 1/n for each of the n elements reduced into output k.

pub(crate) struct MeanDimBackward {
    pub input_shape: Vec<usize>,
    pub n: usize,
}

impl GradFn for MeanDimBackward {
    fn backward(&self, grad_output: &Tensor) -> Vec<Tensor> {
        // grad_output has the kept-dim shape ([.., 1, ..]); spread it back over
        // the reduced dimension and scale by 1/n.
        let spread = broadcast_to(grad_output, &self.input_shape);
        vec![ops::mul_scalar(&spread, 1.0 / self.n as f32)]
    }
}

// --- Clamp: gradient passes through inside (min, max), zero at/outside ---

pub(crate) struct ClampBackward {
    pub input: Tensor,
    pub min: f32,
    pub max: f32,
}

impl GradFn for ClampBackward {
    fn backward(&self, grad_output: &Tensor) -> Vec<Tensor> {
        let data: Vec<f32> = self
            .input
            .data
            .iter()
            .zip(grad_output.data.iter())
            .map(|(&x, &g)| if x > self.min && x < self.max { g } else { 0.0 })
            .collect();
        vec![Tensor::new(data, grad_output.shape().to_vec())]
    }
}

// --- Embedding: gather rows by index; scatter-add gradients on backward ---
// Each output row i was copied from weight row indices[i], so the gradient
// for a weight row is the sum of grad_output rows that looked it up.

pub(crate) struct EmbeddingBackward {
    pub indices: Vec<usize>,
    pub vocab_size: usize,
    pub d_model: usize,
}

impl GradFn for EmbeddingBackward {
    fn backward(&self, grad_output: &Tensor) -> Vec<Tensor> {
        let mut grad_weight = vec![0.0f32; self.vocab_size * self.d_model];
        for (i, &idx) in self.indices.iter().enumerate() {
            let src = i * self.d_model;
            let dst = idx * self.d_model;
            for k in 0..self.d_model {
                grad_weight[dst + k] += grad_output.data[src + k];
            }
        }
        vec![Tensor::new(grad_weight, vec![self.vocab_size, self.d_model])]
    }
}

// --- Matmul: dL/dA = dL/dC @ B^T, dL/dB = A^T @ dL/dC ---

pub(crate) struct MatmulBackward {
    pub a: Tensor,
    pub b: Tensor,
}

impl GradFn for MatmulBackward {
    fn backward(&self, grad_output: &Tensor) -> Vec<Tensor> {
        let grad_a = matmul(grad_output, &self.b.transpose_last2());
        let grad_b = matmul(&self.a.transpose_last2(), grad_output);
        vec![grad_a, grad_b]
    }
}

// --- TransposeLast2: d(transpose_last2(A)) = transpose_last2(grad) ---

pub(crate) struct TransposeLast2Backward;

impl GradFn for TransposeLast2Backward {
    fn backward(&self, grad_output: &Tensor) -> Vec<Tensor> {
        vec![grad_output.transpose_last2()]
    }
}

// --- Reshape: gradient is reshaped back to original shape ---

pub(crate) struct ReshapeBackward {
    pub input_shape: Vec<usize>,
}

impl GradFn for ReshapeBackward {
    fn backward(&self, grad_output: &Tensor) -> Vec<Tensor> {
        vec![grad_output.reshape(&self.input_shape)]
    }
}

// --- SplitHeads: [batch, seq, num_heads*d_k] -> [batch*num_heads, seq, d_k] ---
// Permutation: (b, s, h*d_k+d) -> (b*H+h, s, d)

pub(crate) struct SplitHeadsBackward {
    pub batch: usize,
    pub num_heads: usize,
}

impl GradFn for SplitHeadsBackward {
    fn backward(&self, grad_output: &Tensor) -> Vec<Tensor> {
        // Inverse of split_heads is merge_heads
        vec![merge_heads_tensor(grad_output, self.batch, self.num_heads)]
    }
}

// --- MergeHeads: [batch*num_heads, seq, d_k] -> [batch, seq, num_heads*d_k] ---
// Inverse of SplitHeads

pub(crate) struct MergeHeadsBackward {
    pub num_heads: usize,
}

impl GradFn for MergeHeadsBackward {
    fn backward(&self, grad_output: &Tensor) -> Vec<Tensor> {
        // Inverse of merge_heads is split_heads
        vec![split_heads_tensor(grad_output, self.num_heads)]
    }
}

// Tensor-level split/merge helpers used by both forward and backward

/// [batch, seq, num_heads*d_k] -> [batch*num_heads, seq, d_k]
pub(crate) fn split_heads_tensor(x: &Tensor, num_heads: usize) -> Tensor {
    let shape = x.shape();
    assert_eq!(shape.len(), 3);
    let batch = shape[0];
    let seq = shape[1];
    let d_model = shape[2];
    assert_eq!(d_model % num_heads, 0);
    let d_k = d_model / num_heads;
    let x = x.contiguous();

    let mut out = vec![0.0f32; batch * num_heads * seq * d_k];

    for b in 0..batch {
        for h in 0..num_heads {
            for s in 0..seq {
                for d in 0..d_k {
                    let src = b * (seq * d_model) + s * d_model + h * d_k + d;
                    let dst = (b * num_heads + h) * (seq * d_k) + s * d_k + d;
                    out[dst] = x.data[src];
                }
            }
        }
    }

    Tensor::new(out, vec![batch * num_heads, seq, d_k])
}

/// [batch*num_heads, seq, d_k] -> [batch, seq, num_heads*d_k]
pub(crate) fn merge_heads_tensor(x: &Tensor, batch: usize, num_heads: usize) -> Tensor {
    let shape = x.shape();
    assert_eq!(shape.len(), 3);
    assert_eq!(shape[0], batch * num_heads);
    let seq = shape[1];
    let d_k = shape[2];
    let d_model = num_heads * d_k;
    let x = x.contiguous();

    let mut out = vec![0.0f32; batch * seq * d_model];

    for b in 0..batch {
        for h in 0..num_heads {
            for s in 0..seq {
                for d in 0..d_k {
                    let src = (b * num_heads + h) * (seq * d_k) + s * d_k + d;
                    let dst = b * (seq * d_model) + s * d_model + h * d_k + d;
                    out[dst] = x.data[src];
                }
            }
        }
    }

    Tensor::new(out, vec![batch, seq, d_model])
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

// --- AddScalar: d(x + c)/dx = 1 ---

pub(crate) struct AddScalarBackward;

impl GradFn for AddScalarBackward {
    fn backward(&self, grad_output: &Tensor) -> Vec<Tensor> {
        vec![grad_output.clone()]
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

// --- ReLU: d(relu(x))/dx = 1 if x > 0, else 0 ---

pub(crate) struct ReluBackward {
    pub input: Tensor,
}

impl GradFn for ReluBackward {
    fn backward(&self, grad_output: &Tensor) -> Vec<Tensor> {
        let mask: Vec<f32> = self.input.data.iter()
            .map(|&x| if x > 0.0 { 1.0 } else { 0.0 })
            .collect();
        let mask_t = Tensor::new(mask, self.input.shape().to_vec());
        vec![ops::mul(grad_output, &mask_t)]
    }
}

// --- Sigmoid: d(σ(x))/dx = σ(x) * (1 - σ(x)) ---

pub(crate) struct SigmoidBackward {
    pub output: Tensor,
}

impl GradFn for SigmoidBackward {
    fn backward(&self, grad_output: &Tensor) -> Vec<Tensor> {
        // grad = grad_out * σ * (1 - σ)
        let one_minus_out: Vec<f32> = self.output.data.iter()
            .map(|&s| 1.0 - s)
            .collect();
        let one_minus_t = Tensor::new(one_minus_out, self.output.shape().to_vec());
        let local_grad = ops::mul(&self.output, &one_minus_t);
        vec![ops::mul(grad_output, &local_grad)]
    }
}

// --- Tanh: d(tanh(x))/dx = 1 - tanh²(x) ---

pub(crate) struct TanhBackward {
    pub output: Tensor,
}

impl GradFn for TanhBackward {
    fn backward(&self, grad_output: &Tensor) -> Vec<Tensor> {
        // grad = grad_out * (1 - tanh(x)²)
        let one_minus_sq: Vec<f32> = self.output.data.iter()
            .map(|&t| 1.0 - t * t)
            .collect();
        let local_grad = Tensor::new(one_minus_sq, self.output.shape().to_vec());
        vec![ops::mul(grad_output, &local_grad)]
    }
}

// --- GeLU: fused backward ---
// GeLU(x) = 0.5 * x * (1 + tanh(s))  where s = √(2/π) * (x + 0.044715 * x³)
// GeLU'(x) = 0.5 * (1 + tanh(s)) + 0.5 * x * sech²(s) * s'
// where s' = √(2/π) * (1 + 0.134145 * x²)

pub(crate) struct GeluBackward {
    pub input: Tensor,
}

impl GradFn for GeluBackward {
    fn backward(&self, grad_output: &Tensor) -> Vec<Tensor> {
        const SQRT_2_OVER_PI: f32 = 0.7978845608;
        const C: f32 = 0.044715;
        let local: Vec<f32> = self.input.data.iter().map(|&x| {
            let s = SQRT_2_OVER_PI * (x + C * x * x * x);
            let tanh_s = s.tanh();
            let sech2_s = 1.0 - tanh_s * tanh_s;
            let ds_dx = SQRT_2_OVER_PI * (1.0 + 3.0 * C * x * x);
            0.5 * (1.0 + tanh_s) + 0.5 * x * sech2_s * ds_dx
        }).collect();
        let local_grad = Tensor::new(local, self.input.shape().to_vec());
        vec![ops::mul(grad_output, &local_grad)]
    }
}
