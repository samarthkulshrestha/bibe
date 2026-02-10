use std::cell::RefCell;
use std::collections::HashSet;
use std::rc::Rc;

use crate::tensor::Tensor;
use crate::tensor::ops;
use crate::tensor::stability::stable_softmax;

use super::backward::*;

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
    grad_fn: Option<Box<dyn super::node::GradFn>>,
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
    pub fn from_op(
        tensor: Tensor,
        grad_fn: Box<dyn super::node::GradFn>,
        parents: Vec<Var>,
    ) -> Self {
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

    pub fn div(&self, other: &Var) -> Var {
        let a = self.tensor();
        let b = other.tensor();
        let result = ops::div(&a, &b);
        Var::from_op(
            result,
            Box::new(DivBackward {
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

    pub fn sqrt(&self) -> Var {
        let result = ops::sqrt(&self.tensor());
        Var::from_op(
            result.clone(),
            Box::new(SqrtBackward { output: result.clone() }),
            vec![self.clone()],
        )
    }

    pub fn pow(&self, power: f32) -> Var {
        let input = self.tensor();
        let result = ops::pow(&input, power);
        Var::from_op(
            result,
            Box::new(PowBackward { input, power }),
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

    /// Softmax along a dimension.
    pub fn softmax(&self, dim: usize) -> Var {
        let input = self.tensor();
        let output = stable_softmax(&input, dim);
        Var::from_op(
            output.clone(),
            Box::new(SoftmaxBackward { output: output.clone(), dim }),
            vec![self.clone()],
        )
    }
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
        let loss = a.add(&b).sum();
        loss.backward();

        approx_eq(&a.grad().unwrap().data, &[1.0, 1.0], 1e-6);
        approx_eq(&b.grad().unwrap().data, &[1.0, 1.0], 1e-6);
    }

    #[test]
    fn test_sub_backward() {
        let a = Var::new(Tensor::new(vec![5.0, 3.0], vec![2]), true);
        let b = Var::new(Tensor::new(vec![1.0, 2.0], vec![2]), true);
        let loss = a.sub(&b).sum();
        loss.backward();

        approx_eq(&a.grad().unwrap().data, &[1.0, 1.0], 1e-6);
        approx_eq(&b.grad().unwrap().data, &[-1.0, -1.0], 1e-6);
    }

    #[test]
    fn test_mul_backward() {
        let a = Var::new(Tensor::new(vec![2.0, 3.0], vec![2]), true);
        let b = Var::new(Tensor::new(vec![4.0, 5.0], vec![2]), true);
        let loss = a.mul(&b).sum();
        loss.backward();

        approx_eq(&a.grad().unwrap().data, &[4.0, 5.0], 1e-6);
        approx_eq(&b.grad().unwrap().data, &[2.0, 3.0], 1e-6);
    }

    #[test]
    fn test_div_backward() {
        // f = sum(a / b), a=[10,20], b=[2,5]
        // df/da = 1/b = [0.5, 0.2]
        // df/db = -a/b² = [-10/4, -20/25] = [-2.5, -0.8]
        let a = Var::new(Tensor::new(vec![10.0, 20.0], vec![2]), true);
        let b = Var::new(Tensor::new(vec![2.0, 5.0], vec![2]), true);
        let loss = a.div(&b).sum();
        loss.backward();

        approx_eq(&a.grad().unwrap().data, &[0.5, 0.2], 1e-5);
        approx_eq(&b.grad().unwrap().data, &[-2.5, -0.8], 1e-5);
    }

    #[test]
    fn test_neg_backward() {
        let a = Var::new(Tensor::new(vec![2.0, -3.0], vec![2]), true);
        let loss = a.neg().sum();
        loss.backward();

        approx_eq(&a.grad().unwrap().data, &[-1.0, -1.0], 1e-6);
    }

    #[test]
    fn test_exp_backward() {
        let a = Var::new(Tensor::new(vec![0.0, 1.0], vec![2]), true);
        let loss = a.exp().sum();
        loss.backward();

        let expected = [0.0_f32.exp(), 1.0_f32.exp()];
        approx_eq(&a.grad().unwrap().data, &expected, 1e-5);
    }

    #[test]
    fn test_log_backward() {
        let a = Var::new(Tensor::new(vec![1.0, 2.0, 4.0], vec![3]), true);
        let loss = a.log().sum();
        loss.backward();

        approx_eq(&a.grad().unwrap().data, &[1.0, 0.5, 0.25], 1e-5);
    }

    #[test]
    fn test_sqrt_backward() {
        // f = sum(sqrt(x)), x=[4, 9, 16]
        // df/dx = 1/(2*sqrt(x)) = [1/4, 1/6, 1/8]
        let a = Var::new(Tensor::new(vec![4.0, 9.0, 16.0], vec![3]), true);
        let loss = a.sqrt().sum();
        loss.backward();

        approx_eq(&a.grad().unwrap().data, &[0.25, 1.0 / 6.0, 0.125], 1e-5);
    }

    #[test]
    fn test_pow_backward() {
        // f = sum(x^3), x=[2, 3]
        // df/dx = 3*x^2 = [12, 27]
        let a = Var::new(Tensor::new(vec![2.0, 3.0], vec![2]), true);
        let loss = a.pow(3.0).sum();
        loss.backward();

        approx_eq(&a.grad().unwrap().data, &[12.0, 27.0], 1e-4);
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
        let loss = a.matmul(&b).sum();
        loss.backward();

        approx_eq(
            &a.grad().unwrap().data,
            &[15.0, 19.0, 23.0, 15.0, 19.0, 23.0],
            1e-5,
        );
        approx_eq(
            &b.grad().unwrap().data,
            &[5.0, 5.0, 7.0, 7.0, 9.0, 9.0],
            1e-5,
        );
    }

    #[test]
    fn test_softmax_backward() {
        // softmax grad: s_i * (grad_i - sum(grad * s))
        // For uniform input [0, 0, 0], softmax = [1/3, 1/3, 1/3]
        // With grad_out = [1, 0, 0]:
        //   sum(grad * s) = 1/3
        //   grad_x = s * (grad - sum) = [1/3*(1-1/3), 1/3*(0-1/3), 1/3*(0-1/3)]
        //          = [2/9, -1/9, -1/9]
        let a = Var::new(Tensor::new(vec![0.0, 0.0, 0.0], vec![1, 3]), true);
        let s = a.softmax(1);
        // Pick out first element via mul with one-hot
        let mask = Var::new(Tensor::new(vec![1.0, 0.0, 0.0], vec![1, 3]), false);
        let loss = s.mul(&mask).sum();
        loss.backward();

        approx_eq(
            &a.grad().unwrap().data,
            &[2.0 / 9.0, -1.0 / 9.0, -1.0 / 9.0],
            1e-5,
        );
    }

    #[test]
    fn test_gradient_accumulation() {
        let a = Var::new(Tensor::new(vec![3.0, 5.0], vec![2]), true);
        let loss = a.add(&a).sum();
        loss.backward();
        approx_eq(&a.grad().unwrap().data, &[2.0, 2.0], 1e-6);
    }

    #[test]
    fn test_chain_mul_add() {
        let a = Var::new(Tensor::new(vec![2.0], vec![1]), true);
        let b = Var::new(Tensor::new(vec![3.0], vec![1]), true);
        let c = Var::new(Tensor::new(vec![1.0], vec![1]), true);
        let loss = a.mul(&b).add(&c).sum();
        loss.backward();

        approx_eq(&a.grad().unwrap().data, &[3.0], 1e-6);
        approx_eq(&b.grad().unwrap().data, &[2.0], 1e-6);
        approx_eq(&c.grad().unwrap().data, &[1.0], 1e-6);
    }

    #[test]
    fn test_no_grad_propagation() {
        let a = Var::new(Tensor::new(vec![2.0, 3.0], vec![2]), false);
        let b = Var::new(Tensor::new(vec![4.0, 5.0], vec![2]), true);
        let loss = a.add(&b).sum();
        loss.backward();

        assert!(a.grad().is_none());
        approx_eq(&b.grad().unwrap().data, &[1.0, 1.0], 1e-6);
    }

    #[test]
    fn test_zero_grad() {
        let a = Var::new(Tensor::new(vec![2.0, 3.0], vec![2]), true);
        let b = Var::new(Tensor::new(vec![1.0, 1.0], vec![2]), true);

        a.add(&b).sum().backward();
        approx_eq(&a.grad().unwrap().data, &[1.0, 1.0], 1e-6);

        a.zero_grad();
        b.zero_grad();
        assert!(a.grad().is_none());

        a.mul(&b).sum().backward();
        approx_eq(&a.grad().unwrap().data, &[1.0, 1.0], 1e-6);
    }

    #[test]
    fn test_broadcast_add_backward() {
        let a = Var::new(
            Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]),
            true,
        );
        let b = Var::new(Tensor::new(vec![10.0, 20.0, 30.0], vec![3]), true);
        let loss = a.add(&b).sum();
        loss.backward();

        approx_eq(&a.grad().unwrap().data, &[1.0; 6], 1e-6);
        approx_eq(&b.grad().unwrap().data, &[2.0, 2.0, 2.0], 1e-6);
    }

    #[test]
    fn test_mul_scalar_backward() {
        let a = Var::new(Tensor::new(vec![2.0, 3.0, 4.0], vec![3]), true);
        let loss = a.mul_scalar(5.0).sum();
        loss.backward();
        approx_eq(&a.grad().unwrap().data, &[5.0, 5.0, 5.0], 1e-6);
    }

    #[test]
    fn test_transpose_backward() {
        let a = Var::new(
            Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]),
            true,
        );
        let loss = a.transpose().sum();
        loss.backward();
        approx_eq(&a.grad().unwrap().data, &[1.0; 6], 1e-6);
    }
}
