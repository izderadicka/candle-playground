use candle_core::{D, Device, Result, Tensor};
use candle_nn::{
    Embedding, LayerNorm, LayerNormConfig, Linear, Module, VarBuilder, embedding, layer_norm,
    linear, ops::softmax,
};

pub struct SelfAttention {
    query: Linear,
    key: Linear,
    value: Linear,
    output: Linear,
    scale: f64,
}

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
        let hidden_size = head_size * num_heads;
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
    scale: f64,
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
        let scale = (cfg.hidden_size as f64).sqrt();

        Ok(Self {
            embed,
            pos,
            layers,
            norm_final,
            output,
            scale,
        })
    }

    pub fn forward(&self, x: &Tensor, mask: Option<&Tensor>) -> Result<Tensor> {
        let y = (self.embed.forward(x)? * self.scale)?;
        let mut h = self.pos.forward(&y)?;
        for layer in &self.layers {
            h = layer.forward(&h, mask)?;
        }

        self.output.forward(&self.norm_final.forward(&h)?)
    }
}

fn main() -> Result<()> {
    Ok(())
}
