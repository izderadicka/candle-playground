use std::io::Write as _;

use candle_core::{DType, Device, Result, Tensor};
use candle_datasets::vision::cifar;
use candle_nn::{Linear, Module, Optimizer, VarBuilder, VarMap, linear};

use candle_playground::{batchify, test_classification};

struct Timer {
    start: std::time::Instant,
}

impl Timer {
    fn new() -> Self {
        Self {
            start: std::time::Instant::now(),
        }
    }

    fn report(&self, name: &str) {
        let elapsed = self.start.elapsed();
        println!("{} took {:.2?}", name, elapsed);
    }

    fn r(&mut self, name: &str) {
        self.report(name);
        self.reset();
    }

    fn reset(&mut self) {
        self.start = std::time::Instant::now();
    }

    fn elapsed(&self) -> f32 {
        self.start.elapsed().as_secs_f32()
    }
}

struct Model {
    hidden1: Linear,
    hidden2: Linear,
    hidden3: Linear,
    output: Linear,
}

impl Model {
    fn new(vb: VarBuilder) -> Result<Self> {
        let hidden1 = linear(3 * 32 * 32, 1024, vb.pp("hidden1"))?;
        let hidden2 = linear(1024, 512, vb.pp("hidden2"))?;
        let hidden3 = linear(512, 256, vb.pp("hidden3"))?;
        let output = linear(256, 10, vb.pp("output"))?;
        Ok(Self {
            hidden1,
            hidden2,
            hidden3,
            output,
        })
    }
}

impl Module for Model {
    fn forward(&self, input: &candle_core::Tensor) -> Result<candle_core::Tensor> {
        let x = self.hidden1.forward(input)?;
        let x = x.relu()?;
        let x = self.hidden2.forward(&x)?;
        let x = x.relu()?;
        let x = self.hidden3.forward(&x)?;
        let x = x.relu()?;
        self.output.forward(&x)
    }
}

fn train(train_data: Tensor, train_labels: Tensor, epochs: u32, file_name: &str) -> Result<Model> {
    let device = Device::cuda_if_available(0)?;
    let var_map = VarMap::new();
    let vs = VarBuilder::from_varmap(&var_map, DType::F32, &device);
    let model = Model::new(vs)?;
    let learning_rate = 0.001;
    let mut optimizer = candle_nn::AdamW::new_lr(var_map.all_vars(), learning_rate)?;
    // let mut optimizer = candle_nn::SGD::new(var_map.all_vars(), learning_rate)?;
    for epoch in 0..epochs {
        println!("");
        let timer = Timer::new();
        let mut epoch_loss = 0.0;
        let batches = batchify(&train_data, &train_labels, 1000)?;
        let batches_len = batches.len();
        for (i, (batch_data, batch_labels)) in batches.into_iter().enumerate() {
            let logits = model.forward(&batch_data)?;
            let _log_sm = candle_nn::ops::log_softmax(&logits, candle_core::D::Minus1)?;
            let loss = candle_nn::loss::cross_entropy(&logits, &batch_labels)?;
            optimizer.backward_step(&loss)?;
            epoch_loss += loss.to_scalar::<f32>()?;
            print!(
                "Epoch {epoch} Batch {i}/{batches_len}: Loss = {:.6}\r",
                epoch_loss / (i + 1) as f32
            );
            std::io::stdout().flush().unwrap();
        }
        epoch_loss /= batches_len as f32;
        println!(
            "\nEpoch {epoch}/{epochs}: Loss = {:.6} in {:.2?}",
            epoch_loss,
            timer.elapsed()
        );
    }

    if let Err(e) = var_map.save(file_name) {
        eprintln!("Failed to save model: {e}");
    }

    Ok(model)
}

fn main() -> Result<()> {
    let mut timer = Timer::new();
    let device = candle_core::Device::Cpu;
    let cifar = cifar::load()?;
    let train_images = cifar.train_images.to_device(&device)?.flatten_from(1)?;
    let train_labels = cifar
        .train_labels
        .to_dtype(DType::U32)?
        .to_device(&device)?;
    let test_images = cifar.test_images.to_device(&device)?.flatten_from(1)?;
    let test_labels = cifar.test_labels.to_dtype(DType::U32)?.to_device(&device)?;
    // Experiments with permuting the pixels of the images to see if the model can still learn.
    // let perm = Tensor::from_slice(&permutation(3072), (3072,), &device)?;
    // let train_images = train_images.index_select(&perm, 1)?;
    // let test_images = test_images.index_select(&perm, 1)?; // SAME perm
    println!("CIFAR-10 dataset loaded: {:?}", train_images.shape());
    let train_sample = train_images.get(0)?;
    println!("First training sample: {}", train_sample);
    timer.r("Dataset loading");
    let model = train(
        train_images,
        train_labels,
        60,
        "output/cifar_model.safetensors",
    )?;
    timer.r("Training");
    let (accuracy, _) = test_classification(&model, test_images, test_labels)?;
    println!("Test accuracy: {:.2}%", accuracy * 100.0);
    timer.r("Testing");
    Ok(())
}
