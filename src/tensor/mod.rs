use rand::rng;
use rand_distr::{Distribution, Normal};
use std::ops::{Index, IndexMut};

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

    pub fn hen(shape: &[usize]) -> Self {
        assert!(shape.len() >= 2, "he init requires at least 2D shape");

        let fan_in = shape[shape.len() - 1];
        let std = (2.0 / fan_in as f32).sqrt();

        let mut rng = rng();
        let normal = Normal::new(0.0, std as f64).unwrap();

        let size: usize = shape.iter().product();
        let data: Vec<f32> = (0..size)
            .map(|_| normal.sample(&mut rng) as f32)
            .collect();

        Self::new(data, shape.to_vec())
    }

    pub fn get(&self, indices: &[usize]) -> f32 {
        let flat_idx = self.compute_flat_index(indices);
        self.data[flat_idx]
    }

    pub fn set(&mut self, indices: &[usize], value: f32) {
        let flat_idx = self.compute_flat_index(indices);
        self.data[flat_idx] = value;
    }

    fn compute_flat_index(&self, indices: &[usize]) -> usize {
        assert_eq!(
            indices.len(),
            self.shape.len(),
            "index dimensions {} don't match tensor dimensions {}.",
            indices.len(),
            self.shape.len()
        );

        for (i, (&idx, &dim)) in indices.iter().zip(self.shape.iter()).enumerate() {
            assert!(
                idx < dim,
                "index {} out of bounds for dimension {} (size {}).",
                idx, i, dim
            );
        }

        indices.iter()
            .zip(self.strides.iter())
            .map(|(&idx, &stride)| idx * stride)
            .sum()
    }

    pub fn transpose(&self) -> Self {
        assert_eq!(
            self.shape.len(),
            2,
            "transpose() only works on 2D tensors, got {}D.",
            self.shape.len()
        );

        let new_shape = vec![self.shape[1], self.shape[0]];

        let new_strides = vec![self.strides[1], self.strides[0]];

        Self {
            data: self.data.clone(),
            shape: new_shape,
            strides: new_strides,
            requires_grad: self.requires_grad,
            grad: None,
        }
    }

    pub fn t(&self) -> Self {
        self.transpose()
    }

    pub fn transpose_contiguous(&self) -> Self {
        assert_eq!(self.shape.len(), 2, "transpose only works on 2D tensors.");

        let rows = self.shape[0];
        let cols = self.shape[1];

        let mut new_data = vec![0.0; self.data.len()];

        for i in 0..rows {
            for j in 0..cols {
                let old_idx = i * self.strides[0] + j * self.strides[1];
                let new_idx = j * rows + i;
                new_data[new_idx] = self.data[old_idx];
            }
        }

        Self::new(new_data, vec![cols, rows])
    }
}

impl<const N: usize> Index<[usize; N]> for Tensor {
    type Output = f32;

    fn index(&self, indices: [usize; N]) -> &Self::Output {
        let flat_idx = self.compute_flat_index(&indices);
        &self.data[flat_idx]
    }
}

impl<const N: usize> IndexMut<[usize; N]> for Tensor {
    fn index_mut(&mut self, indices: [usize; N]) -> &mut Self::Output {
        let flat_idx = self.compute_flat_index(&indices);
        &mut self.data[flat_idx]
    }
}
