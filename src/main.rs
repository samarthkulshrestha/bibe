// use std::env;
// use std::error::Error;

fn main() {
    let rn = bibe::tensor::Tensor::randn(&[2, 2]);
    let rn_t = rn.t();

    println!("rn = {:#?}", rn);
    println!("rn_0 = {:#?}", rn[[0, 0]]);
    println!("rn_t = {:#?}", rn_t);
}
