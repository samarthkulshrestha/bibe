use bibe::tensor::{Tensor, matmul::matmul};

fn main() {
    // 2D: [2, 3] @ [3, 4] → [2, 4]
    let a = Tensor::randn(&[2, 3]);
    let b = Tensor::randn(&[3, 4]);
    let c = matmul(&a, &b);
    println!("c = {:#?}", c);

    // Works with transposed tensors
    let bt = Tensor::randn(&[4, 3]);
    let d = matmul(&a, &bt.transpose());  // [2, 3] @ [4, 3].T = [2, 3] @ [3, 4]
    println!("d = {:#?}", d);

    // Batched: [8, 32, 64] @ [8, 64, 32] → [8, 32, 32]
    let q = Tensor::randn(&[8, 32, 64]);
    let k = Tensor::randn(&[8, 64, 32]);
    let attn = matmul(&q, &k);  // attention scores
    println!("attn = {:#?}", attn);
}
