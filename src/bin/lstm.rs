use std::{io::Write, path::PathBuf};

use anyhow::Result;
use candle_core::{DType, Device, IndexOp, Tensor};
use candle_nn::{
    AdamW, Embedding, LSTM, LSTMConfig, Linear, Module, Optimizer, RNN, VarBuilder, VarMap,
    embedding, linear, loss::cross_entropy, lstm,
};
use candle_playground::{
    Timer,
    cli::{Cli, Command},
    text::{Corpus, batch_data, generate_batches},
};
use clap::Parser;
use rand::RngExt;

pub struct Model {
    embed: Embedding,
    lstm: LSTM,
    linear: Linear,
}

impl Model {
    pub fn new(vb: VarBuilder, vocab_size: usize) -> Result<Self> {
        let cfg = LSTMConfig::default();
        let embed = embedding(vocab_size, 64, vb.pp("embed"))?;
        let lstm = lstm(64, 256, cfg, vb.pp("lstm"))?;
        let linear = linear(256, vocab_size, vb.pp("linear"))?;
        Ok(Self {
            embed,
            lstm,
            linear,
        })
    }
}

impl Module for Model {
    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let e = self.embed.forward(&x)?;
        let states = self.lstm.seq(&e)?;
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
    let var_map = VarMap::new();
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
            let (inputs, targets) = batch_data(&batch, dev)?;
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

fn sample(
    model: &Model,
    seed: &str,
    n: usize,
    temp: f64,
    corpus: &Corpus,
    dev: &Device,
    rng: &mut impl RngExt,
) -> Result<String> {
    let mut ctx: Vec<u32> = seed
        .chars()
        .map(|c| corpus.char_to_index(c))
        .map(|r| r.map(|i| i as u32))
        .collect::<Result<Vec<_>>>()?;
    let mut out = String::from(seed);

    for _ in 0..n {
        // one-hot the current context
        let seq_len = ctx.len();

        let input = Tensor::from_slice(&ctx, (1, seq_len), dev)?;

        let logits = model.forward(&input)?; // [1, seq_len, vocab]
        let last = logits.i((0, seq_len - 1))?; // [vocab]
        let probs = candle_nn::ops::softmax(&(last / temp)?, 0)?.to_vec1::<f32>()?;

        // sample from the distribution
        let r: f32 = rng.random();
        let mut acc = 0.0;
        let mut pick = 0u8;
        for (i, p) in probs.iter().enumerate() {
            acc += p;
            if acc >= r {
                pick = i as u8;
                break;
            }
        }

        out.push(corpus.index_to_char(pick)?);
        ctx.push(pick as u32);
        if ctx.len() > 100 {
            ctx.remove(0);
        } // keep context bounded
    }
    Ok(out)
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    let dev = Device::Cpu;
    let mut rng = rand::rng();
    let corpus = Corpus::from_text_file("data/capek.txt")?;
    let vocab_size = corpus.vocab_size();

    match cli.command {
        Command::Train { epochs } => {
            let corpus_indexed: Vec<u8> = corpus.indexed_corpus();

            let training_config = TrainingConfig {
                epochs,
                output_file: Some(cli.file),
                vocab_size,
                window_size: 100,
                batch_size: 64,
                learning_rate: 0.001,
            };

            let _model = train(&corpus_indexed, training_config, &dev, &mut rng)?;
        }
        Command::Sample {
            context,
            size,
            temp,
        } => {
            let mut var_map = VarMap::new();
            let vb = VarBuilder::from_varmap(&var_map, DType::F32, &dev);
            let model = Model::new(vb, vocab_size)?;
            var_map.load(&cli.file)?;
            let output = sample(&model, &context, size, temp, &corpus, &dev, &mut rng)?;
            println!("For context: {context} model generated:");
            println!("{output}");
        }
    }
    Ok(())
}
