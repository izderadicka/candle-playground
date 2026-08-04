use anyhow::Result;
use candle_core::{DType, Device, IndexOp, Tensor};
use candle_nn::{Module, Optimizer, VarBuilder, VarMap};
use rand::seq::SliceRandom;
use rand::{RngExt, rngs};
use rand::{SeedableRng, rngs::StdRng};
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;
use tqdm::tqdm;

// Define hyperparameters
const INPUT_SIZE: usize = 4; // Iris has 4 features
const HIDDEN_SIZE: usize = 32;
const OUTPUT_SIZE: usize = 3; // Iris has 3 classes
const BATCH_SIZE: usize = 32;
const LEARNING_RATE: f64 = 0.01;
const EPOCHS: usize = 500;
const PRINT_EVERY: usize = 10;
const TEST_SPLIT: f32 = 0.2; // 20% for testing

// Simple MLP for Iris classification
struct IrisClassifier {
    layer1: candle_nn::Linear,
    layer2: candle_nn::Linear,
}

impl IrisClassifier {
    fn new(_device: &Device, vb: VarBuilder) -> Result<Self> {
        let layer1 = candle_nn::linear(INPUT_SIZE, HIDDEN_SIZE, vb.pp("layer1"))?;
        let layer2 = candle_nn::linear(HIDDEN_SIZE, OUTPUT_SIZE, vb.pp("layer2"))?;
        Ok(Self { layer1, layer2 })
    }

    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let hidden = self.layer1.forward(input)?;
        let hidden = hidden.relu()?;
        let output = self.layer2.forward(&hidden)?;
        Ok(output)
    }
}

// Load the Iris dataset from file
fn load_iris_dataset(device: &Device) -> Result<((Tensor, Tensor), (Tensor, Tensor))> {
    // Path to the Iris dataset CSV file
    let file_path = Path::new("data/iris.csv");

    // Open the file
    let file = File::open(file_path)?;
    let reader = BufReader::new(file);
    let lines: Vec<String> = reader.lines().collect::<Result<Vec<String>, _>>()?;

    let mut rng = rngs::StdRng::seed_from_u64(42);

    let mut shuffled_lines = lines[1..].to_vec(); // Skip header
    shuffled_lines.shuffle(&mut rng);

    let split_idx = (shuffled_lines.len() as f32 * (1.0 - TEST_SPLIT)) as usize;
    let (train_lines, test_lines) = shuffled_lines.split_at(split_idx);

    let (train_features, train_labels) = prepare_data(train_lines.to_vec(), device)?;
    let (test_features, test_labels) = prepare_data(test_lines.to_vec(), device)?;

    // Normalize features using min-max scaling
    let features_min = train_features.min(0)?.reshape((1, 4))?;
    let features_max = train_features.max(0)?.reshape((1, 4))?;
    let features_range = features_max.sub(&features_min)?;
    let normalized_train_features = train_features
        .broadcast_sub(&features_min)?
        .broadcast_div(&features_range)?;
    let normalized_test_features = test_features
        .broadcast_sub(&features_min)?
        .broadcast_div(&features_range)?;

    Ok((
        (normalized_train_features, train_labels),
        (normalized_test_features, test_labels),
    ))
}

fn prepare_data(lines: Vec<String>, device: &Device) -> Result<(Tensor, Tensor)> {
    // Vectors to store features and labels
    let mut features_data: Vec<f32> = Vec::new();
    let mut labels_data: Vec<u32> = Vec::new();

    // Read the file line by line
    for (i, line) in lines.into_iter().enumerate() {
        let values: Vec<&str> = line.split(',').collect();

        if values.len() != 6 {
            return Err(anyhow::anyhow!(
                "Invalid data format in line {}: {}",
                i,
                line
            ));
        }

        // Parse the 4 feature values
        for j in 1..5 {
            let value = values[j]
                .parse::<f32>()
                .map_err(|_| anyhow::anyhow!("Failed to parse feature value: {}", values[j]))?;
            features_data.push(value);
        }

        // Parse the label (species)
        let label = match values[5] {
            "Iris-setosa" => 0,
            "Iris-versicolor" => 1,
            "Iris-virginica" => 2,
            _ => return Err(anyhow::anyhow!("Unknown species: {}", values[5])),
        };
        labels_data.push(label);
    }

    // Create tensors and normalize features
    let num_samples = labels_data.len();
    let features = Tensor::from_vec(features_data, (num_samples, 4), device)?;
    let labels = Tensor::from_slice(&labels_data, (num_samples,), device)?;
    Ok((features, labels))
}

// Generate batches for training
fn generate_batches(
    features: &Tensor,
    labels: &Tensor,
    batch_size: usize,
    device: &Device,
    rng: &mut StdRng,
) -> Result<Vec<(Tensor, Tensor)>> {
    let num_samples = features.dim(0)?;
    let num_batches = (num_samples + batch_size - 1) / batch_size;

    // Create indices and shuffle them
    let mut indices: Vec<u32> = (0u32..num_samples as u32).collect();
    for i in (1..indices.len()).rev() {
        let j = rng.random_range(0..=i);
        indices.swap(i, j);
    }

    let mut batches = Vec::with_capacity(num_batches);

    for batch_idx in 0..num_batches {
        let start_idx = batch_idx * batch_size;
        let end_idx = std::cmp::min(start_idx + batch_size, num_samples);
        let batch_indices = &indices[start_idx..end_idx];
        let batch_indices_tensor =
            Tensor::from_slice(batch_indices, (batch_indices.len(),), device)?;

        let batch_features_tensor = features.index_select(&batch_indices_tensor, 0)?;
        let batch_labels_tensor = labels.index_select(&batch_indices_tensor, 0)?;
        batches.push((batch_features_tensor, batch_labels_tensor));
    }

    Ok(batches)
}

// Calculate classification accuracy
fn calculate_accuracy(predictions: &Tensor, targets: &Tensor) -> Result<(f32, u32)> {
    let num_samples = targets.dim(0)?;

    let correct = predictions
        .argmax(1)?
        .eq(targets)?
        .to_dtype(DType::F32)?
        .sum_all()?
        .to_scalar::<f32>()?;

    Ok((correct / num_samples as f32, correct as u32))
}

// Build the confusion matrix on-device: rows are true labels, columns are predictions.
fn confusion_matrix(predictions: &Tensor, targets: &Tensor, num_classes: usize) -> Result<Tensor> {
    let device = predictions.device();
    let num_samples = targets.dim(0)?;

    // Fold each (true, pred) pair into a single bin index, then count the bins
    // with a scatter-add. Putting `targets` in the high digit makes rows the true label.
    let stride = Tensor::full(num_classes as u32, (num_samples,), device)?;
    let bins = targets.mul(&stride)?.add(predictions)?;
    let ones = Tensor::ones((num_samples,), DType::U32, device)?;

    let counts = Tensor::zeros((num_classes * num_classes,), DType::U32, device)?
        .index_add(&bins, &ones, 0)?;

    Ok(counts.reshape((num_classes, num_classes))?)
}

fn train() -> Result<()> {
    // Set up device
    let device = Device::new_cuda(0).unwrap_or_else(|_| {
        println!("CUDA device not available, trying Metal...");
        Device::new_metal(0).unwrap_or_else(|_| {
            println!("Metal device not available, falling back to CPU");
            Device::Cpu
        })
    });
    println!("Using device: {:?}", device);

    // Load iris dataset
    let ((features, labels), (test_features, test_labels)) = load_iris_dataset(&device)?;
    println!("Loaded Iris dataset: {} samples", features.dim(0)?);

    // Create model
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let model = IrisClassifier::new(&device, vb)?;

    // Set up optimizer
    let mut optimizer = candle_nn::AdamW::new_lr(varmap.all_vars(), LEARNING_RATE)?;

    // Set up RNG for reproducibility
    let mut rng = StdRng::seed_from_u64(42);

    // Training loop
    println!("Starting training...");
    for epoch in tqdm(0..EPOCHS) {
        // Generate batches
        let batches = generate_batches(&features, &labels, BATCH_SIZE, &device, &mut rng)?;

        let mut epoch_loss = 0.0;
        let mut epoch_correct: u32 = 0;

        for (batch_features, batch_labels) in &batches {
            // Forward pass
            let logits = model.forward(batch_features)?;

            // Calculate loss (cross-entropy)
            let loss = candle_nn::loss::cross_entropy(&logits, batch_labels)?;

            // Backward pass and optimize
            optimizer.backward_step(&loss)?;

            // Calculate accuracy
            let (_batch_accuracy, batch_correct) = calculate_accuracy(&logits, batch_labels)?;

            epoch_loss += loss.to_scalar::<f32>()?;
            epoch_correct += batch_correct;
        }

        epoch_loss /= batches.len() as f32;
        let epoch_accuracy = epoch_correct as f32 / features.dim(0)? as f32;

        // Print epoch summary
        if epoch % PRINT_EVERY == 0 || epoch == EPOCHS - 1 {
            println!(
                "Epoch {}/{}: Loss = {:.4}, Accuracy = {:.4}",
                epoch + 1,
                EPOCHS,
                epoch_loss,
                epoch_accuracy
            );
        }
        if epoch_correct == features.dim(0)? as u32 {
            println!(
                "Early stopping: perfect accuracy achieved at epoch {}",
                epoch + 1
            );
            break;
        }
    }

    // Evaluate on test dataset
    let logits = model.forward(&test_features)?;
    let (accuracy, _correct) = calculate_accuracy(&logits, &test_labels)?;
    println!("\nTest  data classification accuracy: {:.4}", accuracy);

    // Get class predictions
    let predictions = logits.argmax(1)?;

    // Print confusion matrix
    println!("\nConfusion Matrix:");
    let confusion_matrix =
        confusion_matrix(&predictions, &test_labels, OUTPUT_SIZE)?.to_vec2::<u32>()?;

    println!("True\\Pred | Setosa | Versicolor | Virginica");
    println!("----------|--------|------------|----------");
    println!(
        "Setosa    | {:6} | {:10} | {:9}",
        confusion_matrix[0][0], confusion_matrix[0][1], confusion_matrix[0][2]
    );
    println!(
        "Versicolor| {:6} | {:10} | {:9}",
        confusion_matrix[1][0], confusion_matrix[1][1], confusion_matrix[1][2]
    );
    println!(
        "Virginica | {:6} | {:10} | {:9}",
        confusion_matrix[2][0], confusion_matrix[2][1], confusion_matrix[2][2]
    );

    // Print some example predictions
    println!("\nSample predictions:");
    for class_id in 0..OUTPUT_SIZE {
        println!(
            "Class {} ({}): ",
            class_id,
            match class_id {
                0 => "Iris-setosa",
                1 => "Iris-versicolor",
                2 => "Iris-virginica",
                _ => "Unknown",
            }
        );

        let mut count = 0;
        for i in 0..test_features.dim(0)? {
            let true_label = test_labels.i(i)?.to_scalar::<u32>()?;
            let pred_label = predictions.i(i)?.to_scalar::<u32>()?;

            if true_label == class_id as u32 && count < 3 {
                let feature = test_features.i(i)?;
                let feature_vec = feature.to_vec1::<f32>()?;

                println!(
                    "  Sample {}: Features = [{:.2}, {:.2}, {:.2}, {:.2}], Predicted = {}",
                    i,
                    feature_vec[0],
                    feature_vec[1],
                    feature_vec[2],
                    feature_vec[3],
                    match pred_label {
                        0 => "Iris-setosa",
                        1 => "Iris-versicolor",
                        2 => "Iris-virginica",
                        _ => "Unknown",
                    }
                );
                count += 1;
            }
        }
    }

    Ok(())
}

fn main() -> Result<()> {
    train()?;
    Ok(())
}
