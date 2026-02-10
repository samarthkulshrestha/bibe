use crate::tensor::Tensor;

/// Trait for backward functions in the computation graph.
///
/// Each differentiable operation implements this trait to define
/// how gradients flow backward through it. Given the gradient of the
/// output (dL/d_output), it returns gradients for each parent input
/// (dL/d_input_i) in the same order as the parent list.
pub trait GradFn {
    fn backward(&self, grad_output: &Tensor) -> Vec<Tensor>;
}
