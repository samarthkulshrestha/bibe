use rand::rng;
use rand_distr::{Distribution, Normal};

#[derive(Debug, Clone, PartialEq)]
pub struct Tensor {
    data: Vec<f32>,
    shape: Vec<usize>,
    strides: Vec<usize>,
    requires_grad: bool,
    grad: Option<Box<Tensor>>,
}

impl Tensor {
    pub fn new(data: Vec<f32>, shape: Vec<usize>) -> Self {
        assert_eq!(data.len(), shape.iter().product());

        let strides = Self::compute_strides(&shape);
        Self { data, shape, strides, requires_grad: false, grad: None }
    }

    pub fn new_grad(data: Vec<f32>, shape: Vec<usize>,
        requires_grad: bool, grad: Option<Box<Tensor>>)
    -> Self {
        let mut tensor = Self::new(data, shape);
        tensor.requires_grad = requires_grad;
        tensor.grad = grad;
        tensor
    }

    fn compute_strides(shape: &[usize]) -> Vec<usize> {
        let mut strides = vec![1; shape.len()];
        for i in (0..shape.len().saturating_sub(1)).rev() {
            strides[i] = strides[i + 1] * shape[i + 1];
        }
        strides
    }

    pub fn zeros(shape: &[usize]) -> Self {
        let size = shape.iter().product();
        Tensor::new(vec![0.0; size], shape.to_vec())
    }

    pub fn ones(shape: &[usize]) -> Self {
        let size = shape.iter().product();
        Tensor::new(vec![1.0; size], shape.to_vec())
    }

    pub fn randn(shape: &[usize]) -> Self {
        let mut rng = rng();
        let normal = Normal::new(0.0, 1.0).unwrap();

        let size: usize = shape.iter().product();
        let data: Vec<f32> = (0..size)
            .map(|_| normal.sample(&mut rng) as f32)
            .collect();

        Self::new(data, shape.to_vec())
    }

    pub fn xaviern(shape: &[usize]) -> Self {
        assert!(shape.len() >= 2, "xavier init requires at least 2D shape.");

        let fan_in = shape[shape.len() - 1];
        let fan_out = shape[shape.len() - 2];
        let std = (2.0 / (fan_in + fan_out) as f32).sqrt();

        let mut rng = rng();
        let normal = Normal::new(0.0, std as f64).unwrap();

        let size: usize = shape.iter().product();
        let data: Vec<f32> = (0..size)
            .map(|_| normal.sample(&mut rng) as f32)
            .collect();

        Self::new(data, shape.to_vec())
    }

    pub fn xavieru(shape: &[usize]) -> Self {
        assert!(shape.len() >= 2, "xavier init requires at least 2D shape.");

        let fan_in = shape[shape.len() - 1];
        let fan_out = shape[shape.len() - 2];
        let limit = (6.0 / (fan_in + fan_out) as f32).sqrt();

        let mut rng = rng();
        let uniform = rand_distr::Uniform::new(-limit, limit)
            .expect("failed to initialise uniform distr");

        let size: usize = shape.iter().product();
        let data: Vec<f32> = (0..size)
            .map(|_| uniform.sample(&mut rng) as f32)
            .collect();

        Self::new(data, shape.to_vec())
    }
}
