use std::{
    collections::HashMap,
    io::Write,
    path::{Path, PathBuf},
};

use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use candle_nn::{
    AdamW, LSTM, LSTMConfig, Linear, Module, Optimizer, RNN, VarBuilder, VarMap, linear,
    loss::cross_entropy, lstm,
};
use candle_playground::Timer;
use rand::{RngExt, seq::SliceRandom};

pub struct Model {
    lstm: LSTM,
    linear: Linear,
}

impl Model {
    pub fn new(vb: VarBuilder, vocab_size: usize) -> Result<Self> {
        let cfg = LSTMConfig::default();
        let lstm = lstm(vocab_size, 256, cfg, vb.pp("lstm"))?;
        let linear = linear(256, vocab_size, vb.pp("linear"))?;
        Ok(Self { lstm, linear })
    }
}

impl Module for Model {
    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let states = self.lstm.seq(x)?;
        let h = self.lstm.states_to_tensor(&states)?;
        self.linear.forward(&h)
    }
}

struct TrainingConfig {
    epochs: usize,
    output_file: Option<PathBuf>,
    vocab_size: usize,
    window_size: usize,
    batch_size: usize,
    learning_rate: f64,
}

fn train(
    indices: &[u8],
    config: TrainingConfig,
    dev: &Device,
    rng: &mut impl RngExt,
) -> Result<Model> {
    let mut var_map = VarMap::new();
    let vb = VarBuilder::from_varmap(&var_map, DType::F32, dev);
    let model = Model::new(vb, config.vocab_size)?;
    let mut optimizer = AdamW::new_lr(var_map.all_vars(), config.learning_rate)?;
    let epochs = config.epochs;
    println!("");
    for epoch in 0..epochs {
        let epoch_timer = Timer::new();
        let mut epoch_loss = 0.0;
        let batches = generate_batches(indices, config.window_size, config.batch_size, rng)?;
        let num_batches = batches.len();
        for (batch_idx, batch) in batches.iter().enumerate() {
            let (inputs, targets) = batch_data(&batch, config.vocab_size, dev)?;
            let logits = model.forward(&inputs)?;
            let (b, s, v) = logits.dims3()?;
            let loss = cross_entropy(&logits.reshape((b * s, v))?, &targets)?;
            optimizer.backward_step(&loss)?;
            let batch_loss = loss.to_scalar::<f32>()?;
            epoch_loss += batch_loss;
            print!("Epoch {epoch}; Batch {batch_idx}/{num_batches}; Loss {batch_loss:.6};\r");
            std::io::stdout().flush()?;
        }
        let epoch_loss = epoch_loss / num_batches as f32;
        println!(
            "\n Epoch {epoch}/{epochs}; Epoch Loss {epoch_loss:.6}; Took {:.3}",
            epoch_timer.elapsed()
        )
    }

    if let Some(file) = config.output_file {
        if let Err(e) = var_map.save(file) {
            eprint!("Error saving model: {e}")
        }
    }

    Ok(model)
}

fn read_text(file: &Path) -> anyhow::Result<String> {
    std::fs::read_to_string(file).map_err(anyhow::Error::from)
}

fn collect_chars(text: &str) -> HashMap<char, usize> {
    let mut char_counts = HashMap::new();
    for c in text.chars() {
        *char_counts.entry(c).or_insert(0) += 1;
    }
    char_counts
}

fn chars_to_indices(char_counts: &HashMap<char, usize>) -> HashMap<char, u8> {
    let mut chars: Vec<char> = char_counts.keys().cloned().collect();
    chars.sort();
    let mut char_to_index = HashMap::new();
    for (i, c) in chars.iter().enumerate() {
        char_to_index.insert(*c, i.try_into().unwrap());
    }
    char_to_index
}

fn indices_to_chars(char_counts: &HashMap<char, usize>) -> HashMap<u8, char> {
    let mut chars: Vec<char> = char_counts.keys().cloned().collect();
    chars.sort();
    let mut index_to_char = HashMap::new();
    for (i, c) in chars.iter().enumerate() {
        index_to_char.insert(i.try_into().unwrap(), *c);
    }
    index_to_char
}

const WINDOW_SIZE: usize = 100;

type Batches = Vec<Vec<Vec<u8>>>;

fn generate_batches(
    indices: &[u8],
    window_size: usize,
    batch_size: usize,
    rng: &mut impl RngExt,
) -> Result<Batches> {
    let start = rng.random_range(0..WINDOW_SIZE);
    let mut window_starts: Vec<usize> = (start..indices.len() - WINDOW_SIZE)
        .step_by(WINDOW_SIZE)
        .collect();
    window_starts.shuffle(rng);
    let windows = window_starts
        .into_iter()
        .map(|start| &indices[start..start + window_size + 1])
        .map(Vec::from)
        .collect::<Vec<_>>();
    let batches = windows
        .chunks(batch_size)
        .map(Vec::from)
        .collect::<Vec<_>>();

    Ok(batches)
}

fn batch_data(batch: &[Vec<u8>], vocab_size: usize, dev: &Device) -> Result<(Tensor, Tensor)> {
    let batch_size = batch.len();
    let seq_len = batch[0].len() - 1;
    let mut inputs = vec![0f32; batch_size * seq_len * vocab_size]; // one allocation
    let mut targets = Vec::with_capacity(batch_size);

    for (b, window) in batch.iter().enumerate() {
        for (t, &idx) in window[..seq_len].iter().enumerate() {
            inputs[(b * seq_len + t) * vocab_size + idx as usize] = 1.0; // set one element
        }
        for target in window[1..].iter() {
            targets.push(*target as u32);
        }
    }

    let input_tensor = Tensor::from_vec(inputs, (batch_size, seq_len, vocab_size), dev)?;
    let target_tensor =
        Tensor::from_vec(targets, (batch_size * seq_len,), dev)?.to_dtype(DType::U32)?;
    Ok((input_tensor, target_tensor))
}

const VOCAB_SIZE: usize = 112;

fn main() -> anyhow::Result<()> {
    let dev = Device::Cpu;
    let corpus = read_text(Path::new("data/capek.txt"))?;
    let char_counts = collect_chars(&corpus);
    let vocab_size = char_counts.len();
    print!("Corpus length: {}\n", corpus.len());
    print!("Unique characters: {}\n", char_counts.len());
    let chars_map = chars_to_indices(&char_counts);
    let indices_map = indices_to_chars(&char_counts);
    let mut rng = rand::rng();
    let corpus_indexed: Vec<u8> = corpus
        .chars()
        .map(|c| *chars_map.get(&c).unwrap()) // can unwrap as map is done from same data
        .collect();

    let training_config = TrainingConfig {
        epochs: 20,
        output_file: Some(PathBuf::from("output/lstm.safetensors")),
        vocab_size,
        window_size: 100,
        batch_size: 64,
        learning_rate: 0.001,
    };

    let model = train(&corpus_indexed, training_config, &dev, &mut rng)?;
    Ok(())
}
