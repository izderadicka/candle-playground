use std::{io::Write as _, path::PathBuf};

use candle_core::{DType, Device, Result, Tensor};
use candle_datasets::vision::cifar;
use candle_nn::{
    BatchNormConfig, Conv2d, Linear, Module, ModuleT, Optimizer, VarBuilder, VarMap, linear,
};

use candle_playground::{Timer, batchify, test_classification};

struct Model {
    conv1: Conv2d,
    bn1: candle_nn::BatchNorm,
    conv2: Conv2d,
    bn2: candle_nn::BatchNorm,
    conv3: Conv2d,
    bn3: candle_nn::BatchNorm,
    output: Linear,
}

impl Model {
    fn new(vb: VarBuilder) -> Result<Self> {
        let cfg = candle_nn::Conv2dConfig {
            padding: 1,
            ..Default::default()
        };
        let conv1 = candle_nn::conv2d_no_bias(3, 32, 3, cfg, vb.pp("conv1"))?;
        let bn1 = candle_nn::batch_norm(32, BatchNormConfig::default(), vb.pp("bn1"))?;
        let conv2 = candle_nn::conv2d(32, 64, 3, cfg, vb.pp("conv2"))?;
        let bn2 = candle_nn::batch_norm(64, BatchNormConfig::default(), vb.pp("bn2"))?;
        let conv3 = candle_nn::conv2d_no_bias(64, 64, 3, cfg, vb.pp("conv3"))?;
        let bn3 = candle_nn::batch_norm(64, BatchNormConfig::default(), vb.pp("bn3"))?;
        let output = linear(64 * 4 * 4, 10, vb.pp("output"))?;
        Ok(Self {
            conv1,
            bn1,
            conv2,
            bn2,
            conv3,
            bn3,
            output,
        })
    }
}

impl ModuleT for Model {
    fn forward_t(&self, input: &candle_core::Tensor, train: bool) -> Result<candle_core::Tensor> {
        let x = self
            .bn1
            .forward_t(&self.conv1.forward(&input)?, train)?
            .relu()?
            .max_pool2d(2)?;
        let x = self
            .bn2
            .forward_t(&self.conv2.forward(&x)?, train)?
            .relu()?
            .max_pool2d(2)?;
        let x = self
            .bn3
            .forward_t(&self.conv3.forward(&x)?, train)?
            .relu()?
            .max_pool2d(2)?;
        let x = x.flatten_from(1)?;
        self.output.forward(&x)
    }
}

struct TrainingOptions {
    pub epochs: u32,
    pub batch_size: usize,
    pub learning_rate: f64,
    pub show_train_accuracy: bool,
    pub output_file: Option<PathBuf>,
}

impl Default for TrainingOptions {
    fn default() -> Self {
        Self {
            epochs: 15,
            batch_size: 200,
            learning_rate: 0.001,
            show_train_accuracy: false,
            output_file: None,
        }
    }
}

fn train(train_data: Tensor, train_labels: Tensor, opts: TrainingOptions) -> Result<Model> {
    let device = Device::cuda_if_available(0)?;
    let var_map = VarMap::new();
    let vs = VarBuilder::from_varmap(&var_map, DType::F32, &device);
    let model = Model::new(vs)?;
    let learning_rate = opts.learning_rate;
    let epochs = opts.epochs;
    let mut optimizer = candle_nn::AdamW::new_lr(var_map.all_vars(), learning_rate)?;
    // let mut optimizer = candle_nn::SGD::new(var_map.all_vars(), learning_rate)?;
    for epoch in 0..epochs {
        println!("");
        let timer = Timer::new();
        let mut epoch_loss = 0.0;
        let mut epoch_correct = 0.0;
        let batches = batchify(&train_data, &train_labels, opts.batch_size)?;
        let batches_len = batches.len();
        for (i, (batch_data, batch_labels)) in batches.into_iter().enumerate() {
            let logits = model.forward_t(&batch_data, true)?;
            let _log_sm = candle_nn::ops::log_softmax(&logits, candle_core::D::Minus1)?;
            let loss = candle_nn::loss::cross_entropy(&logits, &batch_labels)?;
            optimizer.backward_step(&loss)?;
            epoch_loss += loss.to_scalar::<f32>()?;
            if opts.show_train_accuracy {
                let (_, correct) = test_classification(&model, batch_data, batch_labels)?;
                epoch_correct += correct;
            }
            print!(
                "Epoch {epoch} Batch {i}/{batches_len}: Loss = {:.6}\r",
                epoch_loss / (i + 1) as f32
            );
            std::io::stdout().flush().unwrap();
        }
        epoch_loss /= batches_len as f32;
        let stats = if opts.show_train_accuracy {
            format!(
                "Loss = {:.6}, Accuracy = {:.2}%",
                epoch_loss,
                (epoch_correct / train_data.dim(0)? as f32) * 100.0
            )
        } else {
            format!("Loss = {:.6}", epoch_loss)
        };
        println!(
            "\nEpoch {epoch}/{epochs}: {stats} in {:.2?}",
            timer.elapsed()
        );
    }
    if let Some(file_name) = opts.output_file {
        if let Err(e) = var_map.save(file_name) {
            eprintln!("Failed to save model: {e}");
        }
    }

    Ok(model)
}

fn main() -> Result<()> {
    let mut timer = Timer::new();
    let device = candle_core::Device::Cpu;
    let cifar = cifar::load()?;
    let train_images = cifar.train_images.to_device(&device)?;
    let train_labels = cifar
        .train_labels
        .to_dtype(DType::U32)?
        .to_device(&device)?;
    let test_images = cifar.test_images.to_device(&device)?;
    let test_labels = cifar.test_labels.to_dtype(DType::U32)?.to_device(&device)?;

    println!("CIFAR-10 dataset loaded: {:?}", train_images.shape());
    let train_sample = train_images.get(0)?;
    println!("First training sample: {}", train_sample);
    timer.r("Dataset loading");
    let opts = TrainingOptions {
        epochs: 15,
        batch_size: 200,
        learning_rate: 0.001,
        show_train_accuracy: true,
        output_file: Some(PathBuf::from("output/cifar_model.safetensors")),
    };
    let model = train(train_images, train_labels, opts)?;
    timer.r("Training");
    let (accuracy, _) = test_classification(&model, test_images, test_labels)?;
    println!("Test accuracy: {:.2}%", accuracy * 100.0);
    timer.r("Testing");
    Ok(())
}
