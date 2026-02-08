// use std::env;
// use std::error::Error;

fn main() {
    let rn = bibe::tensor::Tensor::randn(&[2, 2]);
    println!("rn = {:#?}", rn);
}
