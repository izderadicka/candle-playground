use anyhow::Result;
use candle_core::{Device, Tensor};

fn main() -> Result<()> {
    let device = Device::Cpu;

    // Create a 3x1 matrix
    let a = Tensor::new(&[[1f32], [2.], [3.]], &device)?;
    println!("a (3x1): {}", a);

    // Create a 1x4 matrix
    let b = Tensor::new(&[[10f32, 20., 30., 40.]], &device)?;
    println!("b (1x4): {}", b);

    // Broadcasting multiplication
    // a is broadcast to [3, 4] and b is broadcast to [3, 4]
    let result = a.broadcast_mul(&b)?;
    println!("a * b (broadcast to 3x4): {}", result);
    println!("Result shape: {:?}", result.shape());

    // Create a 2x3x1 tensor
    let c = Tensor::new(&[[[1f32], [2.], [3.]], [[4.], [5.], [6.]]], &device)?;
    println!("c (2x3x1): {}", c);

    // Create a 1x1x4 tensor
    let d = Tensor::new(&[[[10f32, 20., 30., 40.]]], &device)?;
    println!("d (1x1x4): {}", d);

    // Broadcasting addition
    // c is broadcast to [2, 3, 4] and d is broadcast to [2, 3, 4]
    let result = c.broadcast_add(&d)?;
    println!("c + d (broadcast to 2x3x4): {}", result);
    println!("Result shape: {:?}", result.shape());

    Ok(())
}
