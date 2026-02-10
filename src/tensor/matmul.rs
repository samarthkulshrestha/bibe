use crate::tensor::Tensor;


/// Matrix multiplication entry point.
/// Supports 2D: [m, k] @ [k, n] → [m, n]
/// Supports 3D (batched): [batch, m, k] @ [batch, k, n] → [batch, m, n]
pub fn matmul(a: &Tensor, b: &Tensor) -> Tensor {
    let a_shape = a.shape();
    let b_shape = b.shape();

    match (a_shape.len(), b_shape.len()) {
        (2, 2) => matmul_2d(a, b),
        (3, 3) => batched_matmul(a, b),
        _ => panic!(
            "matmul requires 2D or 3D tensors, got {}D and {}D",
            a_shape.len(),
            b_shape.len()
        ),
    }
}

/// Naive O(n³) matrix multiplication: [m, k] @ [k, n] → [m, n]
fn matmul_2d(a: &Tensor, b: &Tensor) -> Tensor {
    let a_shape = a.shape();
    let b_shape = b.shape();

    let m = a_shape[0];
    let k = a_shape[1];
    let n = b_shape[1];

    assert_eq!(
        k, b_shape[0],
        "matmul inner dimensions must match: [{}, {}] @ [{}, {}]",
        m, k, b_shape[0], n
    );

    // Make contiguous for direct data access
    let a = a.contiguous();
    let b = b.contiguous();

    let mut out = vec![0.0; m * n];

    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0;
            for p in 0..k {
                sum += a.data[i * k + p] * b.data[p * n + j];
            }
            out[i * n + j] = sum;
        }
    }

    Tensor::new(out, vec![m, n])
}

/// Cache-blocked matmul for better performance on larger matrices.
/// Block size 32 is a reasonable default for most cache hierarchies.
pub fn matmul_blocked(a: &Tensor, b: &Tensor) -> Tensor {
    let a_shape = a.shape();
    let b_shape = b.shape();

    assert_eq!(a_shape.len(), 2, "matmul_blocked requires 2D tensors");
    assert_eq!(b_shape.len(), 2, "matmul_blocked requires 2D tensors");

    let m = a_shape[0];
    let k = a_shape[1];
    let n = b_shape[1];

    assert_eq!(
        k, b_shape[0],
        "matmul inner dimensions must match: [{}, {}] @ [{}, {}]",
        m, k, b_shape[0], n
    );

    let a = a.contiguous();
    let b = b.contiguous();

    let mut out = vec![0.0; m * n];
    const BLOCK: usize = 32;

    for ii in (0..m).step_by(BLOCK) {
        for jj in (0..n).step_by(BLOCK) {
            for pp in (0..k).step_by(BLOCK) {
                let i_end = (ii + BLOCK).min(m);
                let j_end = (jj + BLOCK).min(n);
                let p_end = (pp + BLOCK).min(k);

                for i in ii..i_end {
                    for p in pp..p_end {
                        let a_val = a.data[i * k + p];
                        for j in jj..j_end {
                            out[i * n + j] += a_val * b.data[p * n + j];
                        }
                    }
                }
            }
        }
    }

    Tensor::new(out, vec![m, n])
}

/// Batched matmul: [batch, m, k] @ [batch, k, n] → [batch, m, n]
pub fn batched_matmul(a: &Tensor, b: &Tensor) -> Tensor {
    let a_shape = a.shape();
    let b_shape = b.shape();

    assert_eq!(a_shape.len(), 3, "batched_matmul requires 3D tensors");
    assert_eq!(b_shape.len(), 3, "batched_matmul requires 3D tensors");
    assert_eq!(
        a_shape[0], b_shape[0],
        "batch dimensions must match: {} vs {}",
        a_shape[0], b_shape[0]
    );

    let batch = a_shape[0];
    let m = a_shape[1];
    let k = a_shape[2];
    let n = b_shape[2];

    assert_eq!(
        k, b_shape[1],
        "matmul inner dimensions must match: [{}, {}] @ [{}, {}]",
        m, k, b_shape[1], n
    );

    let a = a.contiguous();
    let b = b.contiguous();

    let slice_a = m * k;
    let slice_b = k * n;
    let slice_o = m * n;

    let mut out = vec![0.0; batch * m * n];

    for bi in 0..batch {
        let a_off = bi * slice_a;
        let b_off = bi * slice_b;
        let o_off = bi * slice_o;

        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0;
                for p in 0..k {
                    sum += a.data[a_off + i * k + p]
                    * b.data[b_off + p * n + j];
                }
                out[o_off + i * n + j] = sum;
            }
        }
    }

    Tensor::new(out, vec![batch, m, n])
}

impl Tensor {
    pub fn matmul(&self, other: &Tensor) -> Tensor {
        matmul(self, other)
    }
}
