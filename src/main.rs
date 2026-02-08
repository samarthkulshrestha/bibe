// use std::env;
// use std::error::Error;

fn main() {
    let rn = bibe::tensor::Tensor::randn(&[2, 2]);
    let xn = bibe::tensor::Tensor::xaviern(&[2, 2]);
    let xu = bibe::tensor::Tensor::xavieru(&[2, 2]);

    println!("rn = {:#?}", rn);
    println!("xn = {:#?}", xn);
    println!("xu = {:#?}", xu);
}
