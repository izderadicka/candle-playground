use candle_core::{DType, Result, Tensor};
use candle_nn::ModuleT;

pub mod cli;
pub mod text;
pub mod token;

pub fn test_classification(
    model: &impl ModuleT,
    test_data: Tensor,
    test_labels: Tensor,
) -> Result<(f32, f32)> {
    let logits = model.forward_t(&test_data, false)?;
    let predictions = logits.argmax(candle_core::D::Minus1)?;
    let correct = predictions
        .eq(&test_labels)?
        .to_dtype(DType::F32)?
        .sum_all()?
        .to_scalar::<f32>()?;
    let accuracy = correct / test_labels.dim(0)? as f32;
    Ok((accuracy, correct))
}

pub fn permutation(n: u32) -> Vec<u32> {
    let mut perm: Vec<u32> = (0..n).collect();
    let mut rng = rand::rng();
    use rand::seq::SliceRandom;
    perm.shuffle(&mut rng);
    perm
}

pub fn batchify(
    data: &Tensor,
    labels: &Tensor,
    batch_size: usize,
) -> Result<Vec<(Tensor, Tensor)>> {
    let indexes = permutation(data.dim(0)?.try_into()?);

    let n_samples = data.dim(0)?;
    let mut batches = Vec::new();
    for i in (0..n_samples).step_by(batch_size) {
        let len = batch_size.min(n_samples - i);
        let batch_indexes = &indexes[i..i + len];
        let batch_indexes = Tensor::from_slice(batch_indexes, (len,), &data.device())?;
        let batch_data = data.index_select(&batch_indexes, 0)?;
        let batch_labels = labels.index_select(&batch_indexes, 0)?;

        batches.push((batch_data, batch_labels));
    }
    Ok(batches)
}

pub struct Timer {
    start: std::time::Instant,
}

impl Timer {
    pub fn new() -> Self {
        Self {
            start: std::time::Instant::now(),
        }
    }

    pub fn report(&self, name: &str) {
        let elapsed = self.start.elapsed();
        println!("{} took {:.2?}", name, elapsed);
    }

    pub fn r(&mut self, name: &str) {
        self.report(name);
        self.reset();
    }

    pub fn reset(&mut self) {
        self.start = std::time::Instant::now();
    }

    pub fn elapsed(&self) -> f32 {
        self.start.elapsed().as_secs_f32()
    }
}
