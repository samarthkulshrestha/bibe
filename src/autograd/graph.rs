use std::cell::RefCell;
use std::collections::HashSet;
use std::rc::Rc;

use crate::tensor::Tensor;
use crate::tensor::broadcast::reduce_sum;
use crate::tensor::ops;

use super::node::GradFn;

/// A tracked variable in the computation graph.
///
/// Wraps a `Tensor` with gradient-tracking metadata. Leaf variables
/// (parameters, inputs) have no `grad_fn`; intermediate results store
/// their producing operation and parent variables so that `backward()`
/// can walk the graph in reverse topological order.
#[derive(Clone)]
pub struct Var {
    inner: Rc<RefCell<VarInner>>,
}

struct VarInner {
    tensor: Tensor,
    grad: Option<Tensor>,
    grad_fn: Option<Box<dyn GradFn>>,
    parents: Vec<Var>,
    requires_grad: bool,
}

impl Var {
    /// Create a leaf variable (parameter or input).
    pub fn new(tensor: Tensor, requires_grad: bool) -> Self {
        Var {
            inner: Rc::new(RefCell::new(VarInner {
                tensor,
                grad: None,
                grad_fn: None,
                parents: vec![],
                requires_grad,
            })),
        }
    }

    /// Create a variable that resulted from an operation.
    /// Gradient tracking is enabled if any parent requires grad.
    pub fn from_op(tensor: Tensor, grad_fn: Box<dyn GradFn>, parents: Vec<Var>) -> Self {
        let requires_grad = parents.iter().any(|p| p.requires_grad());
        Var {
            inner: Rc::new(RefCell::new(VarInner {
                tensor,
                grad: None,
                grad_fn: if requires_grad { Some(grad_fn) } else { None },
                parents,
                requires_grad,
            })),
        }
    }

    /// Access the underlying tensor value.
    pub fn tensor(&self) -> Tensor {
        self.inner.borrow().tensor.clone()
    }

    /// Whether this variable participates in gradient computation.
    pub fn requires_grad(&self) -> bool {
        self.inner.borrow().requires_grad
    }

    /// Retrieve the accumulated gradient, if any.
    pub fn grad(&self) -> Option<Tensor> {
        self.inner.borrow().grad.clone()
    }

    /// Clear the gradient (call before a new forward/backward pass).
    pub fn zero_grad(&self) {
        self.inner.borrow_mut().grad = None;
    }

    /// Run reverse-mode automatic differentiation from this node.
    ///
    /// This should be called on a scalar (single-element) loss tensor.
    /// It seeds dL/dL = 1.0 and propagates gradients to all ancestor
    /// nodes that require grad.
    pub fn backward(&self) {
        {
            let inner = self.inner.borrow();
            let size: usize = inner.tensor.shape().iter().product();
            assert_eq!(
                size, 1,
                "backward() requires a scalar (1-element) tensor, got size {}",
                size
            );
        }

        let topo = self.topological_sort();

        // Seed: dL/dL = 1.0
        let seed_shape = self.inner.borrow().tensor.shape().to_vec();
        self.inner.borrow_mut().grad = Some(Tensor::ones(&seed_shape));

        // Walk in reverse topological order (from loss toward leaves)
        for node in topo.iter().rev() {
            // Extract what we need without holding the borrow
            let (parent_grads, parents) = {
                let inner = node.inner.borrow();
                let grad_output = match &inner.grad {
                    Some(g) => g.clone(),
                    None => continue,
                };
                match &inner.grad_fn {
                    Some(gf) => (gf.backward(&grad_output), inner.parents.clone()),
                    None => continue,
                }
            };

            assert_eq!(
                parent_grads.len(),
                parents.len(),
                "grad_fn returned {} gradients but node has {} parents",
                parent_grads.len(),
                parents.len(),
            );

            // Accumulate gradients into parents
            for (parent, pg) in parents.iter().zip(parent_grads.into_iter()) {
                let mut parent_inner = parent.inner.borrow_mut();
                if !parent_inner.requires_grad {
                    continue;
                }
                match &mut parent_inner.grad {
                    Some(existing) => {
                        *existing = ops::add(existing, &pg);
                    }
                    None => {
                        parent_inner.grad = Some(pg);
                    }
                }
            }
        }
    }

    /// Build a topological ordering of the computation graph via DFS.
    /// Parents appear before children in the returned list.
    fn topological_sort(&self) -> Vec<Var> {
        let mut visited = HashSet::new();
        let mut order = Vec::new();
        self.topo_dfs(&mut visited, &mut order);
        order
    }

    fn topo_dfs(&self, visited: &mut HashSet<usize>, order: &mut Vec<Var>) {
        let ptr = Rc::as_ptr(&self.inner) as usize;
        if visited.contains(&ptr) {
            return;
        }
        visited.insert(ptr);

        let parents: Vec<Var> = self.inner.borrow().parents.clone();
        for parent in &parents {
            parent.topo_dfs(visited, order);
        }
        order.push(self.clone());
    }
}

// ============================================================
// Tracked operations — each builds a graph node with a GradFn
// ============================================================

// --- Add ---

struct AddBackward {
    a_shape: Vec<usize>,
    b_shape: Vec<usize>,
}

impl GradFn for AddBackward {
    fn backward(&self, grad_output: &Tensor) -> Vec<Tensor> {
        let grad_a = reduce_to_shape(grad_output, &self.a_shape);
        let grad_b = reduce_to_shape(grad_output, &self.b_shape);
        vec![grad_a, grad_b]
    }
}

// --- Sub ---

struct SubBackward {
    a_shape: Vec<usize>,
    b_shape: Vec<usize>,
}

impl GradFn for SubBackward {
    fn backward(&self, grad_output: &Tensor) -> Vec<Tensor> {
        let grad_a = reduce_to_shape(grad_output, &self.a_shape);
        let grad_b = reduce_to_shape(&ops::neg(grad_output), &self.b_shape);
        vec![grad_a, grad_b]
    }
}

// --- Mul (element-wise) ---

struct MulBackward {
    a: Tensor,
    b: Tensor,
    a_shape: Vec<usize>,
    b_shape: Vec<usize>,
}

impl GradFn for MulBackward {
    fn backward(&self, grad_output: &Tensor) -> Vec<Tensor> {
        // d(a*b)/da = b, d(a*b)/db = a
        let grad_a = reduce_to_shape(&ops::mul(grad_output, &self.b), &self.a_shape);
        let grad_b = reduce_to_shape(&ops::mul(grad_output, &self.a), &self.b_shape);
        vec![grad_a, grad_b]
    }
}

// --- Neg ---

struct NegBackward;

impl GradFn for NegBackward {
    fn backward(&self, grad_output: &Tensor) -> Vec<Tensor> {
        vec![ops::neg(grad_output)]
    }
}

// --- Exp ---

struct ExpBackward {
    output: Tensor, // exp(x) — reuse the forward output
}

impl GradFn for ExpBackward {
    fn backward(&self, grad_output: &Tensor) -> Vec<Tensor> {
        vec![ops::mul(grad_output, &self.output)]
    }
}

// --- Log ---

struct LogBackward {
    input: Tensor,
}

impl GradFn for LogBackward {
    fn backward(&self, grad_output: &Tensor) -> Vec<Tensor> {
        // d(ln x)/dx = 1/x
        vec![ops::div(grad_output, &self.input)]
    }
}

// --- Sum (reduce to scalar) ---

struct SumBackward {
    input_shape: Vec<usize>,
}

impl GradFn for SumBackward {
    fn backward(&self, grad_output: &Tensor) -> Vec<Tensor> {
        use crate::tensor::broadcast::broadcast_to;
        vec![broadcast_to(grad_output, &self.input_shape)]
    }
}

// --- Matmul ---

struct MatmulBackward {
    a: Tensor,
    b: Tensor,
}

impl GradFn for MatmulBackward {
    fn backward(&self, grad_output: &Tensor) -> Vec<Tensor> {
        use crate::tensor::matmul::matmul;
        // C = A @ B
        // dL/dA = dL/dC @ B^T
        // dL/dB = A^T @ dL/dC
        let grad_a = matmul(grad_output, &self.b.transpose_contiguous());
        let grad_b = matmul(&self.a.transpose_contiguous(), grad_output);
        vec![grad_a, grad_b]
    }
}

// --- MulScalar ---

struct MulScalarBackward {
    scalar: f32,
}

impl GradFn for MulScalarBackward {
    fn backward(&self, grad_output: &Tensor) -> Vec<Tensor> {
        vec![ops::mul_scalar(grad_output, self.scalar)]
    }
}

// --- Transpose ---

struct TransposeBackward;

impl GradFn for TransposeBackward {
    fn backward(&self, grad_output: &Tensor) -> Vec<Tensor> {
        vec![grad_output.transpose_contiguous()]
    }
}

// ============================================================
// Var methods for each tracked operation
// ============================================================

impl Var {
    pub fn add(&self, other: &Var) -> Var {
        let a = self.tensor();
        let b = other.tensor();
        let result = ops::add(&a, &b);
        Var::from_op(
            result,
            Box::new(AddBackward {
                a_shape: a.shape().to_vec(),
                b_shape: b.shape().to_vec(),
            }),
            vec![self.clone(), other.clone()],
        )
    }

    pub fn sub(&self, other: &Var) -> Var {
        let a = self.tensor();
        let b = other.tensor();
        let result = ops::sub(&a, &b);
        Var::from_op(
            result,
            Box::new(SubBackward {
                a_shape: a.shape().to_vec(),
                b_shape: b.shape().to_vec(),
            }),
            vec![self.clone(), other.clone()],
        )
    }

    pub fn mul(&self, other: &Var) -> Var {
        let a = self.tensor();
        let b = other.tensor();
        let result = ops::mul(&a, &b);
        Var::from_op(
            result,
            Box::new(MulBackward {
                a: a.clone(),
                b: b.clone(),
                a_shape: a.shape().to_vec(),
                b_shape: b.shape().to_vec(),
            }),
            vec![self.clone(), other.clone()],
        )
    }

    pub fn neg(&self) -> Var {
        let result = ops::neg(&self.tensor());
        Var::from_op(result, Box::new(NegBackward), vec![self.clone()])
    }

    pub fn exp(&self) -> Var {
        let result = ops::exp(&self.tensor());
        Var::from_op(
            result.clone(),
            Box::new(ExpBackward { output: result.clone() }),
            vec![self.clone()],
        )
    }

    pub fn log(&self) -> Var {
        let input = self.tensor();
        let result = ops::log(&input);
        Var::from_op(
            result,
            Box::new(LogBackward { input }),
            vec![self.clone()],
        )
    }

    /// Sum all elements to a scalar.
    pub fn sum(&self) -> Var {
        let t = self.tensor();
        let input_shape = t.shape().to_vec();
        let total: f32 = t.data.iter().sum();
        let result = Tensor::new(vec![total], vec![1]);
        Var::from_op(
            result,
            Box::new(SumBackward { input_shape }),
            vec![self.clone()],
        )
    }

    /// 2D matrix multiplication.
    pub fn matmul(&self, other: &Var) -> Var {
        let a = self.tensor();
        let b = other.tensor();
        let result = crate::tensor::matmul::matmul(&a, &b);
        Var::from_op(
            result,
            Box::new(MatmulBackward {
                a: a.contiguous(),
                b: b.contiguous(),
            }),
            vec![self.clone(), other.clone()],
        )
    }

    /// Multiply by a scalar constant.
    pub fn mul_scalar(&self, scalar: f32) -> Var {
        let result = ops::mul_scalar(&self.tensor(), scalar);
        Var::from_op(
            result,
            Box::new(MulScalarBackward { scalar }),
            vec![self.clone()],
        )
    }

    /// 2D transpose.
    pub fn transpose(&self) -> Var {
        let result = self.tensor().transpose_contiguous();
        Var::from_op(result, Box::new(TransposeBackward), vec![self.clone()])
    }
}

// ============================================================
// Helper: reduce a gradient to match a (possibly broadcast) shape
// ============================================================

/// When `a` was broadcast from `target_shape` to `grad.shape()` during
/// the forward pass, the backward pass must sum out the broadcast dims
/// to get a gradient of the original shape.
fn reduce_to_shape(grad: &Tensor, target_shape: &[usize]) -> Tensor {
    let grad_shape = grad.shape();
    if grad_shape == target_shape {
        return grad.clone();
    }

    // Pad target_shape on the left with 1s to match grad ndim
    let ndim = grad_shape.len();
    let mut padded = vec![1usize; ndim];
    let offset = ndim.saturating_sub(target_shape.len());
    padded[offset..].copy_from_slice(target_shape);

    // Sum along dimensions that were broadcast (padded dim == 1 but grad dim > 1)
    // Process from highest dim to lowest so indices stay valid after remove
    let mut result = grad.clone();
    for d in (0..ndim).rev() {
        if padded[d] == 1 && result.shape()[d] > 1 {
            result = reduce_sum(&result, d);
        }
    }

    // Reshape to target (removes leading 1s added by broadcast)
    if result.shape() != target_shape {
        result = result.reshape(target_shape);
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: &[f32], b: &[f32], tol: f32) {
        assert_eq!(a.len(), b.len(), "length mismatch: {} vs {}", a.len(), b.len());
        for (i, (&x, &y)) in a.iter().zip(b.iter()).enumerate() {
            assert!(
                (x - y).abs() < tol,
                "element {} differs: {} vs {} (tol={})",
                i, x, y, tol
            );
        }
    }

    #[test]
    fn test_add_backward() {
        let a = Var::new(Tensor::new(vec![2.0, 3.0], vec![2]), true);
        let b = Var::new(Tensor::new(vec![4.0, 5.0], vec![2]), true);
        let c = a.add(&b);
        let loss = c.sum();
        loss.backward();

        approx_eq(&a.grad().unwrap().data, &[1.0, 1.0], 1e-6);
        approx_eq(&b.grad().unwrap().data, &[1.0, 1.0], 1e-6);
    }

    #[test]
    fn test_sub_backward() {
        let a = Var::new(Tensor::new(vec![5.0, 3.0], vec![2]), true);
        let b = Var::new(Tensor::new(vec![1.0, 2.0], vec![2]), true);
        let c = a.sub(&b);
        let loss = c.sum();
        loss.backward();

        approx_eq(&a.grad().unwrap().data, &[1.0, 1.0], 1e-6);
        approx_eq(&b.grad().unwrap().data, &[-1.0, -1.0], 1e-6);
    }

    #[test]
    fn test_mul_backward() {
        let a = Var::new(Tensor::new(vec![2.0, 3.0], vec![2]), true);
        let b = Var::new(Tensor::new(vec![4.0, 5.0], vec![2]), true);
        let c = a.mul(&b);
        let loss = c.sum();
        loss.backward();

        approx_eq(&a.grad().unwrap().data, &[4.0, 5.0], 1e-6);
        approx_eq(&b.grad().unwrap().data, &[2.0, 3.0], 1e-6);
    }

    #[test]
    fn test_neg_backward() {
        let a = Var::new(Tensor::new(vec![2.0, -3.0], vec![2]), true);
        let c = a.neg();
        let loss = c.sum();
        loss.backward();

        approx_eq(&a.grad().unwrap().data, &[-1.0, -1.0], 1e-6);
    }

    #[test]
    fn test_exp_backward() {
        let a = Var::new(Tensor::new(vec![0.0, 1.0], vec![2]), true);
        let c = a.exp();
        let loss = c.sum();
        loss.backward();

        let expected = [0.0_f32.exp(), 1.0_f32.exp()];
        approx_eq(&a.grad().unwrap().data, &expected, 1e-5);
    }

    #[test]
    fn test_log_backward() {
        let a = Var::new(Tensor::new(vec![1.0, 2.0, 4.0], vec![3]), true);
        let c = a.log();
        let loss = c.sum();
        loss.backward();

        approx_eq(&a.grad().unwrap().data, &[1.0, 0.5, 0.25], 1e-5);
    }

    #[test]
    fn test_matmul_backward() {
        let a = Var::new(
            Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]),
            true,
        );
        let b = Var::new(
            Tensor::new(vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0], vec![3, 2]),
            true,
        );
        let c = a.matmul(&b);
        let loss = c.sum();
        loss.backward();

        // dL/dA = ones(2,2) @ B^T
        // B^T = [[7,9,11],[8,10,12]]
        // row of dL/dA = [7+8, 9+10, 11+12] = [15, 19, 23]
        approx_eq(
            &a.grad().unwrap().data,
            &[15.0, 19.0, 23.0, 15.0, 19.0, 23.0],
            1e-5,
        );

        // dL/dB = A^T @ ones(2,2)
        // A^T = [[1,4],[2,5],[3,6]]
        // col of dL/dB = [1+4, 2+5, 3+6] = [5, 7, 9]
        approx_eq(
            &b.grad().unwrap().data,
            &[5.0, 5.0, 7.0, 7.0, 9.0, 9.0],
            1e-5,
        );
    }

    #[test]
    fn test_gradient_accumulation() {
        let a = Var::new(Tensor::new(vec![3.0, 5.0], vec![2]), true);
        let c = a.add(&a); // c = 2a
        let loss = c.sum();
        loss.backward();

        approx_eq(&a.grad().unwrap().data, &[2.0, 2.0], 1e-6);
    }

    #[test]
    fn test_chain_mul_add() {
        let a = Var::new(Tensor::new(vec![2.0], vec![1]), true);
        let b = Var::new(Tensor::new(vec![3.0], vec![1]), true);
        let c = Var::new(Tensor::new(vec![1.0], vec![1]), true);

        let ab = a.mul(&b);
        let abc = ab.add(&c);
        let loss = abc.sum();
        loss.backward();

        approx_eq(&a.grad().unwrap().data, &[3.0], 1e-6);
        approx_eq(&b.grad().unwrap().data, &[2.0], 1e-6);
        approx_eq(&c.grad().unwrap().data, &[1.0], 1e-6);
    }

    #[test]
    fn test_no_grad_propagation() {
        let a = Var::new(Tensor::new(vec![2.0, 3.0], vec![2]), false);
        let b = Var::new(Tensor::new(vec![4.0, 5.0], vec![2]), true);
        let c = a.add(&b);
        let loss = c.sum();
        loss.backward();

        assert!(a.grad().is_none());
        approx_eq(&b.grad().unwrap().data, &[1.0, 1.0], 1e-6);
    }

    #[test]
    fn test_zero_grad() {
        let a = Var::new(Tensor::new(vec![2.0, 3.0], vec![2]), true);
        let b = Var::new(Tensor::new(vec![1.0, 1.0], vec![2]), true);

        let c = a.add(&b);
        let loss = c.sum();
        loss.backward();
        approx_eq(&a.grad().unwrap().data, &[1.0, 1.0], 1e-6);

        a.zero_grad();
        b.zero_grad();
        assert!(a.grad().is_none());

        let c2 = a.mul(&b);
        let loss2 = c2.sum();
        loss2.backward();
        approx_eq(&a.grad().unwrap().data, &[1.0, 1.0], 1e-6);
    }

    #[test]
    fn test_broadcast_add_backward() {
        let a = Var::new(
            Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]),
            true,
        );
        let b = Var::new(Tensor::new(vec![10.0, 20.0, 30.0], vec![3]), true);
        let c = a.add(&b);
        let loss = c.sum();
        loss.backward();

        approx_eq(&a.grad().unwrap().data, &[1.0; 6], 1e-6);
        // b was broadcast over dim 0 (size 2), so grad sums to [2, 2, 2]
        approx_eq(&b.grad().unwrap().data, &[2.0, 2.0, 2.0], 1e-6);
    }

    #[test]
    fn test_mul_scalar_backward() {
        let a = Var::new(Tensor::new(vec![2.0, 3.0, 4.0], vec![3]), true);
        let c = a.mul_scalar(5.0);
        let loss = c.sum();
        loss.backward();

        approx_eq(&a.grad().unwrap().data, &[5.0, 5.0, 5.0], 1e-6);
    }

    #[test]
    fn test_transpose_backward() {
        let a = Var::new(
            Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]),
            true,
        );
        let c = a.transpose();
        let loss = c.sum();
        loss.backward();

        approx_eq(&a.grad().unwrap().data, &[1.0; 6], 1e-6);
    }

    #[test]
    fn test_finite_difference_mul() {
        let eps = 1e-3_f32;
        let a_data = vec![2.0, 3.0, 4.0];
        let b_data = vec![5.0, 6.0, 7.0];

        let a = Var::new(Tensor::new(a_data.clone(), vec![3]), true);
        let b = Var::new(Tensor::new(b_data.clone(), vec![3]), true);
        let loss = a.mul(&b).sum();
        loss.backward();
        let grad_a = a.grad().unwrap();

        for i in 0..3 {
            let mut a_plus = a_data.clone();
            let mut a_minus = a_data.clone();
            a_plus[i] += eps;
            a_minus[i] -= eps;

            let loss_plus: f32 = a_plus.iter().zip(b_data.iter()).map(|(x, y)| x * y).sum();
            let loss_minus: f32 = a_minus.iter().zip(b_data.iter()).map(|(x, y)| x * y).sum();
            let numerical = (loss_plus - loss_minus) / (2.0 * eps);

            assert!(
                (grad_a.data[i] - numerical).abs() < 1e-2,
                "finite diff mismatch at {}: analytical={}, numerical={}",
                i, grad_a.data[i], numerical
            );
        }
    }

    #[test]
    fn test_finite_difference_matmul() {
        let eps = 1e-3_f32;
        let a_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // [2, 3]
        let b_data = vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0]; // [3, 2]

        // Analytical
        let a = Var::new(Tensor::new(a_data.clone(), vec![2, 3]), true);
        let b = Var::new(Tensor::new(b_data.clone(), vec![3, 2]), true);
        let loss = a.matmul(&b).sum();
        loss.backward();
        let grad_a = a.grad().unwrap();

        // Numerical for each element of a
        for i in 0..6 {
            let mut a_plus = a_data.clone();
            let mut a_minus = a_data.clone();
            a_plus[i] += eps;
            a_minus[i] -= eps;

            let c_plus = crate::tensor::matmul::matmul(
                &Tensor::new(a_plus, vec![2, 3]),
                &Tensor::new(b_data.clone(), vec![3, 2]),
            );
            let c_minus = crate::tensor::matmul::matmul(
                &Tensor::new(a_minus, vec![2, 3]),
                &Tensor::new(b_data.clone(), vec![3, 2]),
            );

            let loss_plus: f32 = c_plus.data.iter().sum();
            let loss_minus: f32 = c_minus.data.iter().sum();
            let numerical = (loss_plus - loss_minus) / (2.0 * eps);

            assert!(
                (grad_a.data[i] - numerical).abs() < 1e-1,
                "finite diff mismatch at {}: analytical={}, numerical={}",
                i, grad_a.data[i], numerical
            );
        }
    }
}
