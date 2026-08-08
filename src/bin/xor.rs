use candle_core::{DType, Device, Result, Tensor};
use candle_nn::{Linear, Module, Optimizer, VarBuilder, VarMap, loss, ops};

struct Model {
    first: Linear,
    second: Linear,
}

impl Model {
    fn new(vs: VarBuilder) -> Result<Self> {
        const HIDDEN_WIDTH: usize = 8;
        // let first = make_linear(vs.pp("first"), 2, HIDDEN_WIDTH)?;
        // let second = make_linear(vs.pp("second"), HIDDEN_WIDTH, 1)?;
        let first = candle_nn::linear(2, HIDDEN_WIDTH, vs.pp("first"))?;
        let second = candle_nn::linear(HIDDEN_WIDTH, 1, vs.pp("second"))?;
        Ok(Self { first, second })
    }

    /// Returns probabilities in (0, 1).
    fn predict(&self, input: &Tensor) -> Result<Tensor> {
        ops::sigmoid(&self.forward(input)?)
    }
}

impl Module for Model {
    /// Returns raw logits, as expected by `binary_cross_entropy_with_logit`.
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let x = self.first.forward(input)?;
        let x = x.relu()?;
        self.second.forward(&x)
    }
}

#[allow(dead_code)]
fn make_linear(vs: VarBuilder, in_features: usize, out_features: usize) -> Result<Linear> {
    let ws = vs.get_with_hints(
        (out_features, in_features),
        "weight",
        candle_nn::init::DEFAULT_KAIMING_NORMAL,
    )?;
    let bound = (1.0 / in_features as f64).sqrt();
    let bs = vs.get_with_hints(
        out_features,
        "bias",
        candle_nn::Init::Uniform {
            up: bound,
            lo: -bound,
        },
    )?;
    Ok(Linear::new(ws, Some(bs)))
}

fn make_xor_data(device: &Device) -> Result<(Tensor, Tensor)> {
    let x = Tensor::from_slice(&[0.0f32, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0], (4, 2), device)?;
    let y = Tensor::from_slice(&[0.0f32, 1.0, 1.0, 0.0], (4, 1), device)?;
    Ok((x, y))
}

fn train() -> Result<()> {
    let dev = Device::Cpu;
    let (x, y) = make_xor_data(&dev)?;
    let var_map = VarMap::new();
    let vs = VarBuilder::from_varmap(&var_map, DType::F32, &dev);
    let model = Model::new(vs)?;

    let epochs = 2000;
    let learning_rate = 0.5;
    let mut sgd = candle_nn::SGD::new(var_map.all_vars(), learning_rate)?;

    for epoch in 0..epochs {
        let logits = model.forward(&x)?;
        let loss = loss::binary_cross_entropy_with_logit(&logits, &y)?;
        if epoch % 500 == 0 {
            println!("Epoch {epoch}: loss = {}", loss.to_scalar::<f32>()?);
        }
        sgd.backward_step(&loss)?;
    }

    let probs = model.predict(&x)?;
    println!("predictions: {}", probs.flatten_all()?);

    Ok(())
}

fn main() -> Result<()> {
    train()?;
    Ok(())
}
