use bibe::tensor::{Tensor, ops, broadcast};

fn main() {
    // [3, 1] + [1, 4] → [3, 4]
    let a = Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]);
    let b = Tensor::new(vec![10.0, 20.0, 30.0, 40.0], vec![1, 4]);
    let c = ops::add(&a, &b);
    println!("c = {:#?}", c);
    // [[11, 21, 31, 41],
    //  [12, 22, 32, 42],
    //  [13, 23, 33, 43]]

    // Reduction
    let tensor = Tensor::new(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        vec![2, 3],
    );
    let sum_dim0 = broadcast::reduce_sum(&tensor, 0);  // [5.0, 7.0, 9.0]
    let sum_dim1 = broadcast::reduce_sum(&tensor, 1);  // [6.0, 15.0]
    let mean_dim1 = broadcast::reduce_mean(&tensor, 1); // [2.0, 5.0]
    println!("sum_dim0 = {:#?}", sum_dim0);
    println!("sum_dim1 = {:#?}", sum_dim1);
    println!("mean_dim1 = {:#?}", mean_dim1);
}
