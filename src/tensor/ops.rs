use crate::tensor::{broadcast::{broadcast_shapes, broadcast_to}, Tensor};

// ============================================================
// Binary element-wise operations (require exact shape match)
// ============================================================

/// Element-wise addition with broadcasting: a + b
pub fn add(a: &Tensor, b: &Tensor) -> Tensor {
    let out_shape = broadcast_shapes(a.shape(), b.shape());
    let a_bc = broadcast_to(a, &out_shape);
    let b_bc = broadcast_to(b, &out_shape);

    let data: Vec<f32> = a_bc.data.iter()
        .zip(b_bc.data.iter())
        .map(|(&x, &y)| x + y)
        .collect();

    Tensor::new(data, out_shape)
}

/// Element-wise subtraction: a - b
pub fn sub(a: &Tensor, b: &Tensor) -> Tensor {
    let out_shape = broadcast_shapes(a.shape(), b.shape());
    let a_bc = broadcast_to(a, &out_shape);
    let b_bc = broadcast_to(b, &out_shape);

    let data: Vec<f32> = a_bc.data.iter()
        .zip(b_bc.data.iter())
        .map(|(&x, &y)| x - y)
        .collect();

    Tensor::new(data, out_shape)
}

/// Element-wise multiplication: a * b (Hadamard product)
pub fn mul(a: &Tensor, b: &Tensor) -> Tensor {
    let out_shape = broadcast_shapes(a.shape(), b.shape());
    let a_bc = broadcast_to(a, &out_shape);
    let b_bc = broadcast_to(b, &out_shape);

    let data: Vec<f32> = a_bc.data.iter()
        .zip(b_bc.data.iter())
        .map(|(&x, &y)| x * y)
        .collect();

    Tensor::new(data, out_shape)
}

/// Element-wise division: a / b
pub fn div(a: &Tensor, b: &Tensor) -> Tensor {
    let out_shape = broadcast_shapes(a.shape(), b.shape());
    let a_bc = broadcast_to(a, &out_shape);
    let b_bc = broadcast_to(b, &out_shape);

    let data: Vec<f32> = a_bc.data.iter()
        .zip(b_bc.data.iter())
        .map(|(&x, &y)| x / y)
        .collect();

    Tensor::new(data, out_shape)
}

// ============================================================
// Unary element-wise operations
// ============================================================

/// Element-wise negation: -a
pub fn neg(a: &Tensor) -> Tensor {
    let data: Vec<f32> = a.data.iter().map(|&x| -x).collect();
    Tensor::new(data, a.shape().to_vec())
}

/// Element-wise exponential: e^a
pub fn exp(a: &Tensor) -> Tensor {
    let data: Vec<f32> = a.data.iter().map(|&x| x.exp()).collect();
    Tensor::new(data, a.shape().to_vec())
}

/// Element-wise natural logarithm: ln(a)
/// Uses epsilon for numerical stability: ln(max(x, epsilon))
pub fn log(a: &Tensor) -> Tensor {
    const EPSILON: f32 = 1e-8;
    let data: Vec<f32> = a.data.iter()
        .map(|&x| x.max(EPSILON).ln())
        .collect();
    Tensor::new(data, a.shape().to_vec())
}

/// Element-wise square root: √a
pub fn sqrt(a: &Tensor) -> Tensor {
    let data: Vec<f32> = a.data.iter().map(|&x| x.sqrt()).collect();
    Tensor::new(data, a.shape().to_vec())
}

/// Element-wise power: a^p
pub fn pow(a: &Tensor, p: f32) -> Tensor {
    let data: Vec<f32> = a.data.iter().map(|&x| x.powf(p)).collect();
    Tensor::new(data, a.shape().to_vec())
}

/// Element-wise ReLU: max(0, x)
pub fn relu(a: &Tensor) -> Tensor {
    let data: Vec<f32> = a.data.iter().map(|&x| x.max(0.0)).collect();
    Tensor::new(data, a.shape().to_vec())
}

/// Element-wise sigmoid: 1 / (1 + exp(-x))
pub fn sigmoid(a: &Tensor) -> Tensor {
    let data: Vec<f32> = a.data.iter().map(|&x| {
        if x >= 0.0 {
            1.0 / (1.0 + (-x).exp())
        } else {
            let ex = x.exp();
            ex / (1.0 + ex)
        }
    }).collect();
    Tensor::new(data, a.shape().to_vec())
}

/// Element-wise tanh
pub fn tanh(a: &Tensor) -> Tensor {
    let data: Vec<f32> = a.data.iter().map(|&x| x.tanh()).collect();
    Tensor::new(data, a.shape().to_vec())
}

/// Element-wise GeLU (Gaussian Error Linear Unit):
/// GeLU(x) ≈ 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
pub fn gelu(a: &Tensor) -> Tensor {
    const SQRT_2_OVER_PI: f32 = 0.7978845608; // √(2/π)
    const C: f32 = 0.044715;
    let data: Vec<f32> = a.data.iter().map(|&x| {
        let inner = SQRT_2_OVER_PI * (x + C * x * x * x);
        0.5 * x * (1.0 + inner.tanh())
    }).collect();
    Tensor::new(data, a.shape().to_vec())
}

// ============================================================
// Scalar operations
// ============================================================

/// Add scalar to all elements: a + c
pub fn add_scalar(a: &Tensor, c: f32) -> Tensor {
    let data: Vec<f32> = a.data.iter().map(|&x| x + c).collect();
    Tensor::new(data, a.shape().to_vec())
}

/// Multiply all elements by scalar: a * c
pub fn mul_scalar(a: &Tensor, c: f32) -> Tensor {
    let data: Vec<f32> = a.data.iter().map(|&x| x * c).collect();
    Tensor::new(data, a.shape().to_vec())
}

/// Divide all elements by scalar: a / c
pub fn div_scalar(a: &Tensor, c: f32) -> Tensor {
    let data: Vec<f32> = a.data.iter().map(|&x| x / c).collect();
    Tensor::new(data, a.shape().to_vec())
}

/// Subtract scalar from all elements: a - c
pub fn sub_scalar(a: &Tensor, c: f32) -> Tensor {
    let data: Vec<f32> = a.data.iter().map(|&x| x - c).collect();
    Tensor::new(data, a.shape().to_vec())
}

// ============================================================
// Tensor methods for ergonomics
// ============================================================

impl Tensor {
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn add(&self, other: &Tensor) -> Tensor {
        add(self, other)
    }

    pub fn sub(&self, other: &Tensor) -> Tensor {
        sub(self, other)
    }

    pub fn mul(&self, other: &Tensor) -> Tensor {
        mul(self, other)
    }

    pub fn div(&self, other: &Tensor) -> Tensor {
        div(self, other)
    }

    pub fn neg(&self) -> Tensor {
        neg(self)
    }

    pub fn exp(&self) -> Tensor {
        exp(self)
    }

    pub fn log(&self) -> Tensor {
        log(self)
    }

    pub fn sqrt(&self) -> Tensor {
        sqrt(self)
    }

    pub fn pow(&self, p: f32) -> Tensor {
        pow(self, p)
    }

    pub fn add_scalar(&self, c: f32) -> Tensor {
        add_scalar(self, c)
    }

    pub fn mul_scalar(&self, c: f32) -> Tensor {
        mul_scalar(self, c)
    }

    pub fn div_scalar(&self, c: f32) -> Tensor {
        div_scalar(self, c)
    }

    pub fn sub_scalar(&self, c: f32) -> Tensor {
        sub_scalar(self, c)
    }
}

// ============================================================
// Rust operator overloading for convenience
// ============================================================

use std::ops;

impl ops::Add<&Tensor> for &Tensor {
    type Output = Tensor;
    fn add(self, rhs: &Tensor) -> Tensor {
        add(self, rhs)
    }
}

impl ops::Sub<&Tensor> for &Tensor {
    type Output = Tensor;
    fn sub(self, rhs: &Tensor) -> Tensor {
        sub(self, rhs)
    }
}

impl ops::Mul<&Tensor> for &Tensor {
    type Output = Tensor;
    fn mul(self, rhs: &Tensor) -> Tensor {
        mul(self, rhs)
    }
}

impl ops::Div<&Tensor> for &Tensor {
    type Output = Tensor;
    fn div(self, rhs: &Tensor) -> Tensor {
        div(self, rhs)
    }
}

impl ops::Neg for &Tensor {
    type Output = Tensor;
    fn neg(self) -> Tensor {
        neg(self)
    }
}

// Scalar operations
impl ops::Add<f32> for &Tensor {
    type Output = Tensor;
    fn add(self, rhs: f32) -> Tensor {
        add_scalar(self, rhs)
    }
}

impl ops::Mul<f32> for &Tensor {
    type Output = Tensor;
    fn mul(self, rhs: f32) -> Tensor {
        mul_scalar(self, rhs)
    }
}

impl ops::Div<f32> for &Tensor {
    type Output = Tensor;
    fn div(self, rhs: f32) -> Tensor {
        div_scalar(self, rhs)
    }
}

impl ops::Sub<f32> for &Tensor {
    type Output = Tensor;
    fn sub(self, rhs: f32) -> Tensor {
        sub_scalar(self, rhs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add() {
        let a = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]);
        let b = Tensor::new(vec![4.0, 5.0, 6.0], vec![3]);
        let c = add(&a, &b);
        assert_eq!(c.data, vec![5.0, 7.0, 9.0]);
    }

    #[test]
    fn test_sub() {
        let a = Tensor::new(vec![5.0, 7.0, 9.0], vec![3]);
        let b = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]);
        let c = sub(&a, &b);
        assert_eq!(c.data, vec![4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_mul() {
        let a = Tensor::new(vec![2.0, 3.0, 4.0], vec![3]);
        let b = Tensor::new(vec![5.0, 6.0, 7.0], vec![3]);
        let c = mul(&a, &b);
        assert_eq!(c.data, vec![10.0, 18.0, 28.0]);
    }

    #[test]
    fn test_div() {
        let a = Tensor::new(vec![10.0, 20.0, 30.0], vec![3]);
        let b = Tensor::new(vec![2.0, 4.0, 5.0], vec![3]);
        let c = div(&a, &b);
        assert_eq!(c.data, vec![5.0, 5.0, 6.0]);
    }

    #[test]
    fn test_neg() {
        let a = Tensor::new(vec![1.0, -2.0, 3.0], vec![3]);
        let c = neg(&a);
        assert_eq!(c.data, vec![-1.0, 2.0, -3.0]);
    }

    #[test]
    fn test_exp() {
        let a = Tensor::new(vec![0.0, 1.0, 2.0], vec![3]);
        let c = exp(&a);
        assert!((c.data[0] - 1.0).abs() < 1e-6);
        assert!((c.data[1] - 2.718281828).abs() < 1e-6);
        assert!((c.data[2] - 7.389056099).abs() < 1e-6);
    }

    #[test]
    fn test_log() {
        let a = Tensor::new(vec![1.0, 2.718281828, 7.389056099], vec![3]);
        let c = log(&a);
        assert!((c.data[0] - 0.0).abs() < 1e-6);
        assert!((c.data[1] - 1.0).abs() < 1e-6);
        assert!((c.data[2] - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_log_stability() {
        let a = Tensor::new(vec![0.0, -1.0], vec![2]);
        let c = log(&a);
        // Should use epsilon instead of crashing
        assert!(c.data[0].is_finite());
        assert!(c.data[1].is_finite());
    }

    #[test]
    fn test_sqrt() {
        let a = Tensor::new(vec![1.0, 4.0, 9.0, 16.0], vec![4]);
        let c = sqrt(&a);
        assert_eq!(c.data, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_pow() {
        let a = Tensor::new(vec![2.0, 3.0, 4.0], vec![3]);
        let c = pow(&a, 2.0);
        assert_eq!(c.data, vec![4.0, 9.0, 16.0]);
    }

    #[test]
    fn test_add_scalar() {
        let a = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]);
        let c = add_scalar(&a, 10.0);
        assert_eq!(c.data, vec![11.0, 12.0, 13.0]);
    }

    #[test]
    fn test_mul_scalar() {
        let a = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]);
        let c = mul_scalar(&a, 5.0);
        assert_eq!(c.data, vec![5.0, 10.0, 15.0]);
    }

    #[test]
    fn test_div_scalar() {
        let a = Tensor::new(vec![10.0, 20.0, 30.0], vec![3]);
        let c = div_scalar(&a, 10.0);
        assert_eq!(c.data, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_operator_overloading() {
        let a = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]);
        let b = Tensor::new(vec![4.0, 5.0, 6.0], vec![3]);

        let c = &a + &b;
        assert_eq!(c.data, vec![5.0, 7.0, 9.0]);

        let d = &a * &b;
        assert_eq!(d.data, vec![4.0, 10.0, 18.0]);

        let e = -&a;
        assert_eq!(e.data, vec![-1.0, -2.0, -3.0]);
    }

    #[test]
    fn test_scalar_operator_overloading() {
        let a = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]);

        let c = &a + 10.0;
        assert_eq!(c.data, vec![11.0, 12.0, 13.0]);

        let d = &a * 2.0;
        assert_eq!(d.data, vec![2.0, 4.0, 6.0]);
    }
}
