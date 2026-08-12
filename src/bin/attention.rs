use std::{io::Write, path::PathBuf, sync::Arc};

use candle_core::{D, DType, Device, IndexOp, Result, Tensor};
use candle_nn::{
    AdamW, Embedding, LayerNorm, LayerNormConfig, Linear, Module, Optimizer as _, VarBuilder,
    VarMap, embedding, layer_norm, linear, loss::cross_entropy, ops::softmax,
};
use candle_playground::{
    Timer,
    cli::{Cli, Command},
    token::{self, Batches, Corpus, batch_data, generate_batches},
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

pub struct Rope {
    cos: Tensor,
    sin: Tensor,
}

impl Rope {
    pub fn new(max_seq_length: usize, head_size: usize, dev: &Device) -> Result<Self> {
        let half_size = head_size / 2;
        let mut c = Vec::with_capacity(max_seq_length * half_size);
        let mut s = Vec::with_capacity(max_seq_length * half_size);

        for pos in 0..max_seq_length {
            for i in 0..half_size {
                let theta = pos as f32 / 10_000f32.powf(2.0 * i as f32 / head_size as f32);
                c.push(theta.cos());
                s.push(theta.sin());
            }
        }

        let cos = Tensor::from_vec(c, (max_seq_length, half_size), dev)?;
        let sin = Tensor::from_vec(s, (max_seq_length, half_size), dev)?;

        Ok(Self { sin, cos })
    }

    pub fn apply(&self, x: &Tensor) -> Result<Tensor> {
        let (_b, _heads, seq_size, head_size) = x.dims4()?;
        let half_size = head_size / 2;
        let cos = self
            .cos
            .narrow(0, 0, seq_size)?
            .reshape((1, 1, seq_size, half_size))?;
        let sin = self
            .sin
            .narrow(0, 0, seq_size)?
            .reshape((1, 1, seq_size, half_size))?;

        let x0 = x.narrow(D::Minus1, 0, half_size)?;
        let x1 = x.narrow(D::Minus1, half_size, half_size)?;

        let y0 = (x0.broadcast_mul(&cos)? - x1.broadcast_mul(&sin)?)?;
        let y1 = (x0.broadcast_mul(&sin)? + x1.broadcast_mul(&cos)?)?;

        Tensor::cat(&[&y0, &y1], D::Minus1)
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
    rope: Arc<Rope>,
}

impl MultiHeadAttention {
    pub fn new(
        vb: VarBuilder,
        hidden_size: usize,
        num_heads: usize,
        rope: Arc<Rope>,
    ) -> Result<Self> {
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
            rope,
        })
    }

    pub fn forward(&self, x: &Tensor, mask: Option<&Tensor>) -> Result<Tensor> {
        let (b, s, h) = x.dims3()?;

        let split_heads = |t: Tensor| -> Result<Tensor> {
            t.reshape((b, s, self.num_heads, self.head_size))?
                .transpose(1, 2)?
                .contiguous()
        };
        let q = self.rope.apply(&split_heads(self.query.forward(x)?)?)?; // [b,heads,s,h]
        let k = self.rope.apply(&split_heads(self.key.forward(x)?)?)?;
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

// LayerNorm::forward dispatches to a fused kernel (ops::layer_norm) that has no backward pass
// and silently detaches the autograd graph, so gradients never reach layers below the norm.
// Use the differentiable layer_norm_slow instead.
fn ln_forward(ln: &LayerNorm, x: &Tensor) -> Result<Tensor> {
    candle_nn::ops::layer_norm_slow(x, ln.weight(), ln.bias().unwrap(), ln.eps() as f32)
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
    pub fn new(
        vb: VarBuilder,
        hidden_size: usize,
        num_heads: usize,
        rope: Arc<Rope>,
    ) -> Result<Self> {
        let attention = MultiHeadAttention::new(vb.pp("attention"), hidden_size, num_heads, rope)?;
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
        let h = self.attention.forward(&ln_forward(&self.norm1, x)?, mask)?;
        let x = (x + h)?; // residual path
        let h = self.ff1.forward(&ln_forward(&self.norm2, &x)?)?.relu()?;
        let h = self.ff2.forward(&h)?;
        x + h
    }
}

#[allow(dead_code)]
struct SinPositionalEncoding {
    encoding: Tensor,
}

#[allow(dead_code)]
impl SinPositionalEncoding {
    pub fn new(max_seq_length: usize, hidden_size: usize, device: &Device) -> Result<Self> {
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

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (_b, seq_len, _h) = x.dims3()?;
        let pe = self.encoding.narrow(0, 0, seq_len)?;
        x.broadcast_add(&pe)
    }
}

pub struct Model {
    embed: Embedding,
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
            hidden_size: 256,
            // heads are free - heads * head_size = hidden_size, so the score
            // matmul is the same work either way. head_size 32 keeps 16 rope bands
            num_heads: 8,
            num_layers: 4,
            max_seq_size: 256,
        }
    }
}

impl Model {
    pub fn new(vb: VarBuilder, cfg: ModelConfig, dev: &Device) -> Result<Self> {
        let rope = Arc::new(Rope::new(
            cfg.max_seq_size,
            cfg.hidden_size / cfg.num_heads,
            dev,
        )?);
        let mut layers = Vec::with_capacity(cfg.num_layers);
        for i in 0..cfg.num_layers {
            layers.push(SelfAttentionLayer::new(
                vb.pp(format!("layer{i}")),
                cfg.hidden_size,
                cfg.num_heads,
                rope.clone(),
            )?);
        }
        let embed = embedding(cfg.vocab_size, cfg.hidden_size, vb.pp("embed"))?;
        let norm_final = layer_norm(
            cfg.hidden_size,
            LayerNormConfig::default(),
            vb.pp("norm_final"),
        )?;
        let output = linear(cfg.hidden_size, cfg.vocab_size, vb.pp("output"))?;

        Ok(Self {
            embed,
            layers,
            norm_final,
            output,
        })
    }

    pub fn forward(&self, x: &Tensor, mask: Option<&Tensor>) -> Result<Tensor> {
        let mut h = self.embed.forward(x)?;
        for layer in &self.layers {
            h = layer.forward(&h, mask)?;
        }

        self.output.forward(&ln_forward(&self.norm_final, &h)?)
    }
}

struct TrainingConfig {
    epochs: usize,
    output_file: PathBuf,
    model_cfg: ModelConfig,
    window_size: usize,
    batch_size: usize,
    learning_rate: f64,
    checkpoint: Option<PathBuf>,
    checkpoint_every: usize,
}

/// Contiguous windows from offset 0, no shuffling and no random start - the
/// validation number has to mean the same thing from one run to the next.
///
/// Empty if there are not enough tokens for a single window.
fn validation_batches(tokens: &[u32], window_size: usize, batch_size: usize) -> Batches {
    tokens
        .windows(window_size + 1)
        .step_by(window_size)
        .map(Vec::from)
        .collect::<Vec<_>>()
        .chunks(batch_size)
        .map(Vec::from)
        .collect()
}

/// Mean loss over the held out batches - forward only, no backward step.
/// `None` when there is nothing to validate on.
fn evaluate(
    model: &Model,
    batches: &Batches,
    causal_mask: &Tensor,
    dev: &Device,
) -> anyhow::Result<Option<f32>> {
    if batches.is_empty() {
        return Ok(None);
    }
    let mut total = 0.0;
    for batch in batches {
        let (inputs, targets) = batch_data(batch, dev)?;
        let logits = model.forward(&inputs, Some(causal_mask))?;
        let (b, s, v) = logits.dims3()?;
        total += cross_entropy(&logits.reshape((b * s, v))?, &targets)?.to_scalar::<f32>()?;
    }
    Ok(Some(total / batches.len() as f32))
}

fn train(
    indices: &[u32],
    validation: &[u32],
    config: TrainingConfig,
    dev: &Device,
    rng: &mut impl RngExt,
) -> anyhow::Result<Model> {
    let mut var_map = VarMap::new();
    let vb = VarBuilder::from_varmap(&var_map, DType::F32, dev);
    let model = Model::new(vb, config.model_cfg, dev)?;
    if let Some(checkpoint) = config.checkpoint {
        var_map.load(&checkpoint)?;
    }
    let mut optimizer = AdamW::new_lr(var_map.all_vars(), config.learning_rate)?;
    let epochs = config.epochs;
    let causal_mask = causal_mask(config.window_size, dev)?;
    let mut cpt_file = config.output_file.clone();
    cpt_file.add_extension("ckp");
    let mut best_file = config.output_file.clone();
    best_file.add_extension("best");

    // Tiled from offset 0 without shuffling, unlike the training batches, so the
    // validation loss is comparable across runs and not just across epochs.
    let val_batches = validation_batches(validation, config.window_size, config.batch_size);
    let mut best_loss = f32::INFINITY;
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

        let val_loss = evaluate(&model, &val_batches, &causal_mask, dev)?;
        match val_loss {
            Some(l) => println!(
                "\n Epoch {epoch}/{epochs}; Train Loss {epoch_loss:.6}; Val Loss {l:.6}; Took {:.3}",
                epoch_timer.elapsed()
            ),
            None => println!(
                "\n Epoch {epoch}/{epochs}; Train Loss {epoch_loss:.6}; Took {:.3}",
                epoch_timer.elapsed()
            ),
        }

        // every epoch by default - at half an hour each, losing one hurts
        if config.checkpoint_every > 0 && epoch % config.checkpoint_every == 0 {
            if let Err(e) = var_map.save(&cpt_file) {
                eprintln!("Error saving checkpoint: {e}");
            }
        }
        // the last model is not the best one once validation loss turns back up
        if let Some(l) = val_loss
            && l < best_loss
        {
            best_loss = l;
            if let Err(e) = var_map.save(&best_file) {
                eprintln!("Error saving best model: {e}");
            }
        }
    }

    if let Err(e) = var_map.save(config.output_file) {
        eprint!("Error saving model: {e}")
    }

    Ok(model)
}

struct SamplingConfig {
    temp: f64,
    top_k: Option<usize>,
    top_p: Option<f32>,
    model_file: PathBuf,
    max_context_size: usize,
    model_cfg: ModelConfig,
}

impl Default for SamplingConfig {
    fn default() -> Self {
        Self {
            temp: 0.7,
            top_k: Some(10),
            top_p: Some(0.9),
            model_file: Default::default(),
            max_context_size: 100,
            model_cfg: Default::default(),
        }
    }
}

fn apply_selection(
    probs: Vec<f32>,
    top_k: Option<usize>,
    top_p: Option<f32>,
    rng: &mut impl RngExt,
) -> u32 {
    let mut indices: Vec<_> = (0..probs.len()).collect();
    indices.sort_by(|a, b| probs[*b].total_cmp(&probs[*a]));
    if let Some(k) = top_k {
        indices.truncate(k);
    }
    if let Some(p) = top_p {
        let mut acc_p = 0.0;
        let mut idx = 0;
        while idx < indices.len() {
            acc_p += probs[indices[idx]];
            if acc_p >= p {
                break;
            }
            idx += 1
        }
        indices.truncate(idx + 1)
    }
    let sum: f32 = indices.iter().map(|i| probs[*i]).sum();

    // sample from the distribution
    let r: f32 = rng.random();
    let mut acc = 0.0;
    let mut pick = indices.first().map(|i| u32::try_from(*i).unwrap()).unwrap();
    for i in indices {
        let p = probs[i] / sum;
        acc += p;
        if acc >= r {
            pick = i.try_into().unwrap();
            break;
        }
    }

    pick
}

fn sample(
    seed: &str,
    n: usize,
    config: SamplingConfig,
    corpus: &Corpus,
    dev: &Device,
    rng: &mut impl RngExt,
) -> anyhow::Result<String> {
    let mut var_map = VarMap::new();
    let vb = VarBuilder::from_varmap(&var_map, DType::F32, &dev);
    let model = Model::new(vb, config.model_cfg, &dev)?;
    var_map.load(&config.model_file)?;
    let mut ctx: Vec<u32> = corpus.encode(seed)?;
    if ctx.is_empty() {
        anyhow::bail!("Seed encodes to no tokens - give a non empty context");
    }
    if ctx.len() > config.max_context_size {
        ctx.drain(..ctx.len() - config.max_context_size);
    }
    // ctx is the sliding window fed to the model, generated is everything we
    // produced - BPE tokens carry their own spacing, so the whole run is
    // decoded in one go at the end rather than token by token
    let mut generated = ctx.clone();

    for _ in 0..n {
        // one-hot the current context
        let seq_len = ctx.len();
        let causal_mask = causal_mask(seq_len, dev)?;
        let input = Tensor::from_slice(&ctx, (1, seq_len), dev)?;

        let logits = model.forward(&input, Some(&causal_mask))?; // [1, seq_len, vocab]
        let last = logits.i((0, seq_len - 1))?; // [vocab]
        let probs = candle_nn::ops::softmax(&(last / config.temp)?, 0)?.to_vec1::<f32>()?;

        let pick = apply_selection(probs, config.top_k, config.top_p, rng);

        generated.push(pick);
        ctx.push(pick);
        if ctx.len() > config.max_context_size {
            ctx.remove(0);
        } // keep context bounded
    }
    corpus.decode(&generated)
}

//best sampling performace is achieved when training window size is same as max context in sampling
// now counted in tokens, not characters
const WINDOW_SIZE: usize = 200;
/// Fraction of the corpus held out to watch for memorisation
const VALIDATION_SPLIT: f64 = 0.05;
const CORPUS_FILE: &str = "data/capek.txt";

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    let dev = Device::Cpu;
    let mut rng = rand::rng();

    match cli.command {
        Command::Train {
            model,
            epochs,
            checkpoint,
        } => {
            let corpus = Corpus::from_files(&cli.tokenizer, CORPUS_FILE)?;
            let model_cfg = ModelConfig {
                vocab_size: corpus.vocab_size(),
                ..Default::default()
            };

            // held out as a contiguous tail, so no training window overlaps it
            let tokens = corpus.tokens();
            let split = tokens.len() - (tokens.len() as f64 * VALIDATION_SPLIT) as usize;
            let (train_tokens, val_tokens) = tokens.split_at(split);
            println!(
                "Training on {} tokens, validating on {}",
                train_tokens.len(),
                val_tokens.len()
            );

            let training_config = TrainingConfig {
                epochs,
                output_file: model,
                model_cfg,
                window_size: WINDOW_SIZE,
                batch_size: 32,
                learning_rate: 0.001,
                checkpoint,
                checkpoint_every: 1,
            };

            let _model = train(train_tokens, val_tokens, training_config, &dev, &mut rng)?;
        }
        Command::Sample {
            model,
            context,
            size,
            temp,
            top_k,
            top_p,
        } => {
            // sampling only needs the tokenizer, not the corpus text
            let corpus = Corpus::from_tokenizer_file(&cli.tokenizer)?;
            let model_cfg = ModelConfig {
                vocab_size: corpus.vocab_size(),
                ..Default::default()
            };
            let cfg = SamplingConfig {
                temp,
                top_k,
                top_p,
                model_cfg,
                model_file: model,
                max_context_size: WINDOW_SIZE,
            };
            let output = sample(&context, size, cfg, &corpus, &dev, &mut rng)?;
            println!("For context: {context} model generated:");
            println!("{output}");
        }
        Command::Tokenize {
            num_tokens,
            corpus,
            chars,
        } => {
            let train = if chars {
                token::train_chars
            } else {
                token::train
            };
            train(&corpus, num_tokens, &cli.tokenizer).map_err(|e| anyhow::anyhow!(e))?;
        }
    }
    Ok(())
}
