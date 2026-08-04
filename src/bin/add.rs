use core::error;

use candle_core::{DType, Device, Result, Tensor, Var};
use candle_nn::{Linear, Module, Optimizer, VarBuilder, VarMap, linear};
use rand::{Rng, RngExt};

struct Model {
    first: Linear,
    second: Linear,
}

impl Model {
    pub fn new(vb: VarBuilder) -> Result<Self> {
        let first = linear(2, 16, vb.pp("first"))?;
        let second = linear(16, 1, vb.pp("second"))?;
        Ok(Self { first, second })
    }

    pub fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let y = self.first.forward(&input)?;
        let y = y.relu()?;
        self.second.forward(&y)
    }
}

fn generate_data(batch_size: usize, dev: &Device) -> Result<(Tensor, Tensor)> {
    let mut rng = rand::rng();

    let mut inputs = Vec::with_capacity(batch_size * 2);
    let mut targets = Vec::with_capacity(batch_size);

    for _ in 0..batch_size {
        let a: f32 = rng.random_range(0.0..=NUM_RANGE);
        let b: f32 = rng.random_range(0.0..=NUM_RANGE);
        let y = a + b;
        inputs.push(a);
        inputs.push(b);
        targets.push(y);
    }

    let inputs = Tensor::from_slice(&inputs, (batch_size, 2), dev)?;
    let targets = Tensor::from_slice(&targets, (batch_size, 1), dev)?;

    Ok((inputs, targets))
}

fn generate_data_oob(batch_size: usize, dev: &Device) -> Result<(Tensor, Tensor)> {
    let mut rng = rand::rng();

    let mut inputs = Vec::with_capacity(batch_size * 2);
    let mut targets = Vec::with_capacity(batch_size);

    for _ in 0..batch_size {
        let a: f32 = rng.random_range(NUM_RANGE..=10.0 * NUM_RANGE);
        let b: f32 = rng.random_range(NUM_RANGE..=10.0 * NUM_RANGE);
        let y = a + b;
        inputs.push(a);
        inputs.push(b);
        targets.push(y);
    }

    let inputs = Tensor::from_slice(&inputs, (batch_size, 2), dev)?;
    let targets = Tensor::from_slice(&targets, (batch_size, 1), dev)?;

    Ok((inputs, targets))
}

fn train(dev: &Device) -> Result<Model> {
    let var_map = VarMap::new();
    let vb = VarBuilder::from_varmap(&var_map, DType::F32, dev);
    let model = Model::new(vb)?;

    let (batch_size, epochs) = (64, 1000);
    let lr = 0.1;
    let (x, y) = generate_data(batch_size, dev)?;
    let mut opt = candle_nn::AdamW::new_lr(var_map.all_vars(), lr)?;
    for epoch in 0..epochs {
        let mut batch_loss = 0.0;
        const BATCHES: usize = 20;
        for _ in 0..BATCHES {
            let y_pred = model.forward(&x)?;
            let loss = candle_nn::loss::mse(&y_pred, &y)?;
            opt.backward_step(&loss)?;
            batch_loss += loss.to_scalar::<f32>()?;
        }

        if epoch % 100 == 0 {
            println!("epoch {epoch:5}: loss = {:.6}", batch_loss / BATCHES as f32);
        }
    }

    Ok(model)
}

fn test(data: &[(f32, f32)], model: &Model, dev: &Device) -> Result<()> {
    for (a, b) in data {
        let x = Tensor::from_slice(&[*a, *b], (1, 2), dev)?;
        let y_pred = model.forward(&x)?;
        let y_pred = y_pred.reshape(())?.to_scalar::<f32>()?;
        let y_correct = a + b;
        let error = (y_pred - y_correct).abs();
        println!("{a} + {b} = {y_pred:.3}; error = {error:.3}");
    }

    Ok(())
}

const NUM_RANGE: f32 = 10.0;
fn main() -> Result<()> {
    let dev = Device::Cpu;
    let model = train(&dev)?;
    println!("Testing on in-range data:");
    let test_cases = [
        (3.0, 5.0),
        (2.5, 7.5),
        (1.2, 3.4),
        (8.0, 9.0),
        (0.0, 0.0),
        (NUM_RANGE, NUM_RANGE), // Test edge case
    ];
    test(test_cases.as_slice(), &model, &dev)?;
    println!("Testing on out-of-range data:");
    let test_data2 = [
        (NUM_RANGE + 1.0, NUM_RANGE + 2.0),
        (NUM_RANGE + 3.0, NUM_RANGE + 4.0),
        (NUM_RANGE + 5.0, NUM_RANGE + 6.0),
    ];
    test(&test_data2, &model, &dev)?;

    println!("Testing on random out-of-range data:");
    let (x, _) = generate_data_oob(5, &dev)?;
    let test_data3: Vec<(f32, f32)> = x
        .to_vec2::<f32>()?
        .into_iter()
        .map(|row| (row[0], row[1]))
        .collect();
    test(&test_data3, &model, &dev)?;

    Ok(())
}
