use candle_core::{D, DType, Device, IndexOp, Result, Tensor};
use candle_nn::{Linear, Module, Optimizer, VarBuilder, VarMap, loss, ops};
use clap::{Parser, Subcommand};

#[derive(Parser, Debug)]
#[command(name = "candle-playground")]
struct Args {
    #[arg(long, default_value = "output/mnist_model.safetensors")]
    model_file: String,
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand, Debug)]
enum Command {
    Train {
        #[arg(long, default_value_t = 10)]
        epochs: u32,
    },
    Infer {
        #[arg(value_name = "IMAGE")]
        image: String,
    },
}

struct Model {
    first: Linear,
    second: Linear,
}

impl Model {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let x = self.first.forward(input)?;
        let x = x.relu()?;
        self.second.forward(&x)
    }

    fn new(vs: VarBuilder) -> Result<Self> {
        const IMAGE_DIM: usize = 784;
        const HIDDEN_DIM: usize = 100;
        const OUTPUT_DIM: usize = 10;
        let first = make_linear(vs.pp("first"), IMAGE_DIM, HIDDEN_DIM)?;
        let second = make_linear(vs.pp("second"), HIDDEN_DIM, OUTPUT_DIM)?;
        Ok(Self { first, second })
    }
}

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

// fn random_layer(in_features: usize, out_features: usize, device: &Device) -> Result<Linear> {
//     let weight = Tensor::randn(0f32, 1.0, (out_features, in_features), device)?;
//     let bias = Tensor::randn(0f32, 1.0, (out_features,), device)?;
//     Ok(Linear::new(weight, Some(bias)))
// }

fn infer(input: Tensor, file_name: &str) -> Result<u8> {
    let dev = Device::cuda_if_available(0)?;
    let mut var_map = VarMap::new();

    let vs = VarBuilder::from_varmap(&var_map, DType::F32, &dev);
    let model = Model::new(vs)?;
    var_map.load(file_name)?;

    let logits = model.forward(&input)?;
    let logits = logits.i(0)?;
    let predicted_class = logits.argmax(D::Minus1)?.to_scalar::<u32>()?;
    Ok(predicted_class as u8)
}
fn train(data: candle_datasets::vision::Dataset, epochs: u32, file_name: &str) -> Result<Model> {
    let dev = Device::cuda_if_available(0)?;

    let train_images = data.train_images.to_device(&dev)?;
    let train_labels = data.train_labels.to_dtype(DType::U32)?.to_device(&dev)?;
    let test_images = data.test_images.to_device(&dev)?;
    let test_labels = data.test_labels.to_dtype(DType::U32)?.to_device(&dev)?;

    let file_path = std::path::Path::new(file_name);

    let mut var_map = VarMap::new();
    let vs = VarBuilder::from_varmap(&var_map, DType::F32, &dev);
    let model = Model::new(vs.clone())?;

    if file_path.exists() {
        var_map.load(file_name)?;
    }

    let learning_rate = 0.05;

    let mut sgd = candle_nn::SGD::new(var_map.all_vars(), learning_rate)?;
    for epoch in 0..epochs {
        let logits = model.forward(&train_images)?;
        let log_sm = ops::log_softmax(&logits, D::Minus1)?;

        let loss = loss::nll(&log_sm, &train_labels)?;
        sgd.backward_step(&loss)?;

        let test_logits = model.forward(&test_images)?;
        let sum_ok = test_logits
            .argmax(D::Minus1)?
            .eq(&test_labels)?
            .to_dtype(DType::F32)?
            .sum_all()?
            .to_scalar::<f32>()?;
        let test_accuracy = sum_ok / test_labels.dims1()? as f32;
        println!(
            "{epoch:4} train loss: {:8.5} test acc: {:5.2}%",
            loss.to_scalar::<f32>()?,
            test_accuracy
        );
    }
    var_map.save(file_path)?;
    Ok(model)
}

fn main() -> Result<()> {
    // let device = Device::Cpu;
    // let first = random_layer(768, 100, &device)?;
    // let second = random_layer(100, 10, &device)?;
    // let model = Model { first, second };
    // let dummy_image = Tensor::randn(0f32, 1.0, (1, 768), &device)?;
    // let output = model.forward(&dummy_image)?;

    // println!("Output: {output}");
    let args = Args::parse();

    match args.command {
        Command::Train { epochs } => {
            let data = candle_datasets::vision::mnist::load()?;
            let _model = train(data, epochs, &args.model_file)?;
        }
        Command::Infer { image } => {
            let data = candle_datasets::vision::mnist::load()?;
            let dev = Device::cuda_if_available(0)?;
            for index in 0..data.test_images.dim(0)? {
                let one_image = data.test_images.i(index)?.to_vec1::<f32>()?;
                let label = data.test_labels.i(index)?.to_scalar::<u8>()?;

                let input = data
                    .test_images
                    .i(index)?
                    .to_dtype(DType::F32)?
                    .to_device(&dev)?;
                let input = input.unsqueeze(0)?;
                let predicted_class = infer(input, &args.model_file)?;
                if label != predicted_class {
                    println!(
                        "Mismatch at index {index}: label {label}, predicted {predicted_class}"
                    );

                    print_image(&one_image);
                    // wait for user input to continue
                    println!("Press Enter to continue...");
                    std::io::stdin().read_line(&mut String::new()).unwrap();
                }
            }
        }
    }

    Ok(())
}

fn print_image(image: &[f32]) {
    for row in 0..28 {
        for col in 0..28 {
            let pixel = image[row * 28 + col];
            if pixel > 0.5 {
                print!("#");
            } else if pixel > 0.1 {
                print!(".");
            } else {
                print!(" ");
            }
        }
        println!();
    }
}
