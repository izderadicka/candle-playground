use candle_core::{DType, Device, Result, Tensor};

fn sigmoid(z: &Tensor) -> Result<Tensor> {
    // 1 / (1 + exp(-z))
    (z.neg()?.exp()? + 1.0)?.recip()
}

fn main() -> Result<()> {
    let dev = Device::Cpu;
    let n = 4usize;
    let h = 8usize;
    let lr = 0.5;

    let x = Tensor::from_slice(&[0f32, 0., 0., 1., 1., 0., 1., 1.], (4, 2), &dev)?;
    let y = Tensor::from_slice(&[0f32, 1., 1., 0.], (4, 1), &dev)?;

    // Parameters are plain Tensors, NOT Vars — we update them ourselves.
    let mut w1 = (Tensor::randn(0f32, 1.0, (2, h), &dev)? * (2.0f64 / 2.0).sqrt())?;
    let mut b1 = Tensor::zeros(h, DType::F32, &dev)?;
    let mut w2 = (Tensor::randn(0f32, 1.0, (h, h), &dev)? * (2.0f64 / h as f64).sqrt())?;
    let mut b2 = Tensor::zeros(h, DType::F32, &dev)?;
    let mut w3 = (Tensor::randn(0f32, 1.0, (h, 1), &dev)? * (2.0f64 / h as f64).sqrt())?;
    let mut b3 = Tensor::zeros(1, DType::F32, &dev)?;

    fn forward(
        x: &Tensor,
        w1: &Tensor,
        w2: &Tensor,
        w3: &Tensor,
        b1: &Tensor,
        b2: &Tensor,
        b3: &Tensor,
    ) -> Result<Tensor> {
        let z1 = x.matmul(&w1)?.broadcast_add(&b1)?; // (4,8)
        let h1 = z1.relu()?; // (4,8)
        let z2 = h1.matmul(&w2)?.broadcast_add(&b2)?;
        let h2 = z2.relu()?; // (4,8)
        let z3 = h2.matmul(&w3)?.broadcast_add(&b3)?;
        sigmoid(&z3)
    }

    for epoch in 0..3000 {
        // ---- forward ----
        let z1 = x.matmul(&w1)?.broadcast_add(&b1)?; // (4,8)
        let h1 = z1.relu()?; // (4,8)
        let z2 = h1.matmul(&w2)?.broadcast_add(&b2)?;
        let h2 = z2.relu()?; // (4,8)
        let z3 = h2.matmul(&w3)?.broadcast_add(&b3)?;
        let p = sigmoid(&z3)?; // (4,1)

        // ---- backward, by hand ----
        // w3
        let dz3 = ((&p - &y)? / n as f64)?; // (4,1)
        let dw3 = h2.t()?.matmul(&dz3)?; // (8,1)
        let db3 = dz3.sum(0)?; // (1,)
        let dh2 = dz3.matmul(&w3.t()?)?;
        // w2
        // relu': mask of where z1 was positive
        let mask = z2.gt(&z2.zeros_like()?)?.to_dtype(DType::F32)?;
        let dz2 = (dh2 * mask)?; // (4,8)
        let dw2 = h1.t()?.matmul(&dz2)?; // (2,8)
        let db2 = dz2.sum(0)?;
        let dh1 = dz2.matmul(&w2.t()?)?; // (8,)

        //w1
        let mask = z1.gt(&z1.zeros_like()?)?.to_dtype(DType::F32)?;
        let dz1 = (dh1 * mask)?; // (4,8)
        let dw1 = x.t()?.matmul(&dz1)?; // (2,8)
        let db1 = dz1.sum(0)?;

        // ---- SGD update: p ← p − lr·∇p ----
        w1 = (&w1 - (dw1 * lr)?)?;
        b1 = (&b1 - (db1 * lr)?)?;
        w2 = (&w2 - (dw2 * lr)?)?;
        b2 = (&b2 - (db2 * lr)?)?;
        w3 = (&w3 - (dw3 * lr)?)?;
        b3 = (&b3 - (db3 * lr)?)?;

        if epoch % 500 == 0 {
            let loss = candle_nn::loss::binary_cross_entropy_with_logit(&z3, &y)?;
            println!("epoch {epoch:5}: loss = {:.6}", loss.to_scalar::<f32>()?);
        }
    }

    let p = forward(&x, &w1, &w2, &w3, &b1, &b2, &b3)?;
    println!("predictions:\n{:?}", p.flatten_all()?);
    Ok(())
}
