pub mod stability;
pub mod matmul;
pub mod broadcast;
pub mod ops;

use rand_distr::{Distribution, Normal};
use std::ops::{Index, IndexMut};

use crate::rng::with_rng;

#[derive(Debug, Clone, PartialEq)]
pub struct Tensor {
    pub data: Vec<f32>,
    shape: Vec<usize>,
    strides: Vec<usize>,
}

impl Tensor {
    pub fn new(data: Vec<f32>, shape: Vec<usize>) -> Self {
        assert_eq!(data.len(), shape.iter().product());

        let strides = Self::compute_strides(&shape);
        Self { data, shape, strides }
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
        let normal = Normal::new(0.0, 1.0).unwrap();
        let size: usize = shape.iter().product();
        let data: Vec<f32> =
            with_rng(|rng| (0..size).map(|_| normal.sample(rng) as f32).collect());
        Self::new(data, shape.to_vec())
    }

    pub fn xaviern(shape: &[usize]) -> Self {
        assert!(shape.len() >= 2, "xavier init requires at least 2D shape.");

        let fan_in = shape[shape.len() - 1];
        let fan_out = shape[shape.len() - 2];
        let std = (2.0 / (fan_in + fan_out) as f32).sqrt();

        let normal = Normal::new(0.0, std as f64).unwrap();
        let size: usize = shape.iter().product();
        let data: Vec<f32> =
            with_rng(|rng| (0..size).map(|_| normal.sample(rng) as f32).collect());
        Self::new(data, shape.to_vec())
    }

    pub fn xavieru(shape: &[usize]) -> Self {
        assert!(shape.len() >= 2, "xavier init requires at least 2D shape.");

        let fan_in = shape[shape.len() - 1];
        let fan_out = shape[shape.len() - 2];
        let limit = (6.0 / (fan_in + fan_out) as f32).sqrt();

        let uniform = rand_distr::Uniform::new(-limit, limit)
            .expect("failed to initialise uniform distr");
        let size: usize = shape.iter().product();
        let data: Vec<f32> =
            with_rng(|rng| (0..size).map(|_| uniform.sample(rng)).collect());
        Self::new(data, shape.to_vec())
    }

    pub fn hen(shape: &[usize]) -> Self {
        assert!(shape.len() >= 2, "he init requires at least 2D shape");

        let fan_in = shape[shape.len() - 1];
        let std = (2.0 / fan_in as f32).sqrt();

        let normal = Normal::new(0.0, std as f64).unwrap();
        let size: usize = shape.iter().product();
        let data: Vec<f32> =
            with_rng(|rng| (0..size).map(|_| normal.sample(rng) as f32).collect());
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

    /// Transpose the last two dimensions. Works for 2D and 3D+ tensors.
    /// For 2D: equivalent to transpose_contiguous.
    /// For 3D [batch, m, n] -> [batch, n, m].
    pub fn transpose_last2(&self) -> Self {
        let ndim = self.shape.len();
        assert!(ndim >= 2, "transpose_last2 requires at least 2D tensor, got {}D", ndim);

        if ndim == 2 {
            return self.transpose_contiguous();
        }

        let t = self.contiguous();
        let batch_dims: Vec<usize> = t.shape[..ndim - 2].to_vec();
        let m = t.shape[ndim - 2];
        let n = t.shape[ndim - 1];
        let batch_size: usize = batch_dims.iter().product();
        let slice = m * n;

        let mut new_data = vec![0.0; t.data.len()];

        for bi in 0..batch_size {
            let off = bi * slice;
            for i in 0..m {
                for j in 0..n {
                    new_data[off + j * m + i] = t.data[off + i * n + j];
                }
            }
        }

        let mut new_shape = batch_dims;
        new_shape.push(n);
        new_shape.push(m);
        Self::new(new_data, new_shape)
    }

    pub fn reshape(&self, new_shape: &[usize]) -> Self {
        let old_size: usize = self.shape.iter().product();
        let new_size: usize = new_shape.iter().product();

        assert_eq!(
            old_size, new_size,
            "cannot reshape tensor of size {} to size {}.",
            old_size, new_size
        );

        if !self.is_contiguous() {
            panic!(
                "cannot reshape non-contiguous tensor. call .contiguous() first.\n  \
                    shape: {:?}, strides: {:?}",
                self.shape, self.strides
            );
        }

        Self {
            data: self.data.clone(),
            shape: new_shape.to_vec(),
            strides: Self::compute_strides(new_shape),
        }
    }

    pub fn is_contiguous(&self) -> bool {
        if self.shape.is_empty() {
            return true;
        }

        let expected_strides = Self::compute_strides(&self.shape);
        self.strides == expected_strides
    }

    pub fn contiguous(&self) -> Self {
        if self.is_contiguous() {
            return self.clone();
        }

        let size: usize = self.shape.iter().product();
        let mut new_data = Vec::with_capacity(size);

        self.iterate_indices(|indices| {
            new_data.push(self.get(indices));
        });

        Self::new(new_data, self.shape.clone())
    }

    fn iterate_indices<F>(&self, mut f: F) where F: FnMut(&[usize]),
    {
        let mut indices = vec![0; self.shape.len()];
        let size: usize = self.shape.iter().product();

        for _ in 0..size {
            f(&indices);

            for i in (0..indices.len()).rev() {
                indices[i] += 1;
                if indices[i] < self.shape[i] {
                    break;
                }
                indices[i] = 0;
            }
        }
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
