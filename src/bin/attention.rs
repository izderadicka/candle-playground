use std::{io::Write, path::PathBuf};

use candle_core::{D, DType, Device, IndexOp, Result, Tensor};
use candle_nn::{
    AdamW, Embedding, LayerNorm, LayerNormConfig, Linear, Module, Optimizer as _, VarBuilder,
    VarMap, embedding, layer_norm, linear, loss::cross_entropy, ops::softmax,
};
use candle_playground::{
    Timer,
    cli::{Cli, Command},
    text::{Corpus, batch_data, generate_batches},
};
use clap::Parser as _;
use rand::RngExt;

pub struct SelfAttention {
    query: Linear,
    key: Linear,
    value: Linear,
    output: Linear,
    scale: f64,
}

#[allow(dead_code)]
impl SelfAttention {
    pub fn new(vb: VarBuilder, hidden_size: usize) -> Result<Self> {
        let query = linear(hidden_size, hidden_size, vb.pp("query"))?;
        let key = linear(hidden_size, hidden_size, vb.pp("key"))?;
        let value = linear(hidden_size, hidden_size, vb.pp("value"))?;
        let output = linear(hidden_size, hidden_size, vb.pp("output"))?;
        let scale = 1.0 / (hidden_size as f64).sqrt();
        Ok(Self {
            query,
            key,
            value,
            output,
            scale,
        })
    }

    pub fn forward(&self, x: &Tensor, mask: Option<&Tensor>) -> Result<Tensor> {
        let q = self.query.forward(x)?; // [b,s,h]
        let k = self.key.forward(x)?;
        let v = self.value.forward(x)?;

        let scores = (q.matmul(&k.transpose(1, 2)?)? * self.scale)?; // [b,s,s]
        let scores = match mask {
            Some(m) => scores.broadcast_add(m)?,
            None => scores,
        };

        let weights = softmax(&scores, D::Minus1)?;
        let context = weights.matmul(&v)?; //[b,s,h]
        self.output.forward(&context) // [b,s,h]
    }
}

pub struct MultiHeadAttention {
    query: Linear,
    key: Linear,
    value: Linear,
    output: Linear,
    num_heads: usize,
    head_size: usize, // one head size = embedding size
    scale: f64,
}

impl MultiHeadAttention {
    pub fn new(vb: VarBuilder, hidden_size: usize, num_heads: usize) -> Result<Self> {
        assert!(hidden_size % num_heads == 0);
        let head_size = hidden_size / num_heads;
        let query = linear(hidden_size, hidden_size, vb.pp("query"))?;
        let key = linear(hidden_size, hidden_size, vb.pp("key"))?;
        let value = linear(hidden_size, hidden_size, vb.pp("value"))?;
        let output = linear(hidden_size, hidden_size, vb.pp("output"))?;
        let scale = 1.0 / (head_size as f64).sqrt();
        Ok(Self {
            query,
            key,
            value,
            output,
            scale,
            head_size,
            num_heads,
        })
    }

    pub fn forward(&self, x: &Tensor, mask: Option<&Tensor>) -> Result<Tensor> {
        let (b, s, h) = x.dims3()?;

        let split_heads = |t: Tensor| -> Result<Tensor> {
            t.reshape((b, s, self.num_heads, self.head_size))?
                .transpose(1, 2)?
                .contiguous()
        };
        let q = split_heads(self.query.forward(x)?)?; // [b,heads,s,h]
        let k = split_heads(self.key.forward(x)?)?;
        let v = split_heads(self.value.forward(x)?)?;

        let scores = (q.matmul(&k.transpose(2, 3)?.contiguous()?)? * self.scale)?; // [b,heads,s,s]
        let scores = match mask {
            Some(m) => scores.broadcast_add(m)?,
            None => scores,
        };

        let weights = softmax(&scores, D::Minus1)?;
        let context = weights.matmul(&v)?; // [b, heads, s, h]
        let context = context.transpose(1, 2)?.reshape((b, s, h))?;
        self.output.forward(&context) // [b,s,h]
    }
}

fn causal_mask(seq_len: usize, dev: &Device) -> Result<Tensor> {
    let mut m = vec![0f32; seq_len * seq_len];
    for i in 0..seq_len {
        for j in (i + 1)..seq_len {
            m[i * seq_len + j] = f32::NEG_INFINITY; // or -1e9 if causes problem
        }
    }

    Tensor::from_vec(m, (1, 1, seq_len, seq_len), dev)
}

struct SelfAttentionLayer {
    attention: MultiHeadAttention,
    norm1: LayerNorm,
    ff1: Linear,
    ff2: Linear,
    norm2: LayerNorm,
}

impl SelfAttentionLayer {
    pub fn new(vb: VarBuilder, hidden_size: usize, num_heads: usize) -> Result<Self> {
        let attention = MultiHeadAttention::new(vb.pp("attention"), hidden_size, num_heads)?;
        let norm_config = LayerNormConfig::default();
        let norm1 = layer_norm(hidden_size, norm_config, vb.pp("norm1"))?;
        let ff1 = linear(hidden_size, 4 * hidden_size, vb.pp("ff1"))?;
        let ff2 = linear(4 * hidden_size, hidden_size, vb.pp("ff2"))?;
        let norm2 = layer_norm(hidden_size, norm_config, vb.pp("norm2"))?;

        Ok(Self {
            attention,
            norm1,
            ff1,
            ff2,
            norm2,
        })
    }

    pub fn forward(&self, x: &Tensor, mask: Option<&Tensor>) -> Result<Tensor> {
        let h = self.attention.forward(&self.norm1.forward(x)?, mask)?;
        let x = (x + h)?; // residual path
        let h = self.ff1.forward(&self.norm2.forward(&x)?)?.relu()?;
        let h = self.ff2.forward(&h)?;
        x + h
    }
}

struct SinPositionalEncoding {
    encoding: Tensor,
}

impl SinPositionalEncoding {
    fn new(max_seq_length: usize, hidden_size: usize, device: &Device) -> Result<Self> {
        // Create positional encoding matrix
        let mut encoding = vec![0.0; max_seq_length * hidden_size];

        for pos in 0..max_seq_length {
            for i in 0..hidden_size {
                let div_term = 10000.0_f32.powf(2.0 * (i / 2) as f32 / hidden_size as f32);
                if i % 2 == 0 {
                    encoding[pos * hidden_size + i] = (pos as f32 / div_term).sin();
                } else {
                    encoding[pos * hidden_size + i] = (pos as f32 / div_term).cos();
                }
            }
        }

        let encoding = Tensor::from_slice(&encoding, (max_seq_length, hidden_size), device)?;

        Ok(Self { encoding })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (_b, seq_len, _h) = x.dims3()?;
        let pe = self.encoding.narrow(0, 0, seq_len)?;
        x.broadcast_add(&pe)
    }
}

pub struct Model {
    embed: Embedding,
    pos: SinPositionalEncoding, // Positional encoding
    layers: Vec<SelfAttentionLayer>,
    norm_final: LayerNorm,
    output: Linear,
}

pub struct ModelConfig {
    vocab_size: usize,
    hidden_size: usize,
    num_heads: usize,
    num_layers: usize,
    max_seq_size: usize,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            vocab_size: 122,
            hidden_size: 128,
            num_heads: 4,
            num_layers: 4,
            max_seq_size: 256,
        }
    }
}

impl Model {
    pub fn new(vb: VarBuilder, cfg: ModelConfig, dev: &Device) -> Result<Self> {
        let mut layers = Vec::with_capacity(cfg.num_layers);
        for i in 0..cfg.num_layers {
            layers.push(SelfAttentionLayer::new(
                vb.pp(format!("layer{i}")),
                cfg.hidden_size,
                cfg.num_heads,
            )?);
        }
        let embed = embedding(cfg.vocab_size, cfg.hidden_size, vb.pp("embed"))?;
        let pos = SinPositionalEncoding::new(cfg.max_seq_size, cfg.hidden_size, dev)?;
        let norm_final = layer_norm(
            cfg.hidden_size,
            LayerNormConfig::default(),
            vb.pp("norm_final"),
        )?;
        let output = linear(cfg.hidden_size, cfg.vocab_size, vb.pp("output"))?;

        Ok(Self {
            embed,
            pos,
            layers,
            norm_final,
            output,
        })
    }

    pub fn forward(&self, x: &Tensor, mask: Option<&Tensor>) -> Result<Tensor> {
        let y = self.embed.forward(x)?;
        let mut h = self.pos.forward(&y)?;
        for layer in &self.layers {
            h = layer.forward(&h, mask)?;
        }

        self.output.forward(&self.norm_final.forward(&h)?)
    }
}

struct TrainingConfig {
    epochs: usize,
    output_file: Option<PathBuf>,
    model_cfg: ModelConfig,
    window_size: usize,
    batch_size: usize,
    learning_rate: f64,
}

fn train(
    indices: &[u8],
    config: TrainingConfig,
    dev: &Device,
    rng: &mut impl RngExt,
) -> anyhow::Result<Model> {
    let var_map = VarMap::new();
    let vb = VarBuilder::from_varmap(&var_map, DType::F32, dev);
    let model = Model::new(vb, config.model_cfg, dev)?;
    let mut optimizer = AdamW::new_lr(var_map.all_vars(), config.learning_rate)?;
    let epochs = config.epochs;
    let causal_mask = causal_mask(config.window_size, dev)?;
    println!("");
    for epoch in 0..epochs {
        let epoch_timer = Timer::new();
        let mut epoch_loss = 0.0;
        let batches = generate_batches(indices, config.window_size, config.batch_size, rng)?;
        let num_batches = batches.len();
        for (batch_idx, batch) in batches.iter().enumerate() {
            let (inputs, targets) = batch_data(&batch, dev)?;
            let logits = model.forward(&inputs, Some(&causal_mask))?;
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
) -> anyhow::Result<String> {
    let mut ctx: Vec<u32> = seed
        .chars()
        .map(|c| corpus.char_to_index(c))
        .map(|r| r.map(|i| i as u32))
        .collect::<anyhow::Result<Vec<_>>>()?;
    let mut out = String::from(seed);

    for _ in 0..n {
        // one-hot the current context
        let seq_len = ctx.len();
        let causal_mask = causal_mask(seq_len, dev)?;
        let input = Tensor::from_slice(&ctx, (1, seq_len), dev)?;

        let logits = model.forward(&input, Some(&causal_mask))?; // [1, seq_len, vocab]
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
    let model_cfg = ModelConfig {
        vocab_size,
        ..Default::default()
    };

    match cli.command {
        Command::Train { epochs } => {
            let corpus_indexed: Vec<u8> = corpus.indexed_corpus();

            let training_config = TrainingConfig {
                epochs,
                output_file: Some(cli.file),
                model_cfg,
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
            let model = Model::new(vb, model_cfg, &dev)?;
            var_map.load(&cli.file)?;
            let output = sample(&model, &context, size, temp, &corpus, &dev, &mut rng)?;
            println!("For context: {context} model generated:");
            println!("{output}");
        }
    }
    Ok(())
}
