use clap::{Parser, Subcommand};
use std::path::PathBuf;

#[derive(Parser)]
pub struct Cli {
    /// tokenizer.json - written by `tokenize`, read by `train` and `sample`
    #[arg(
        short,
        long,
        global = true,
        default_value = "data/capek-tokens.json",
        help = "tokenizer file"
    )]
    pub tokenizer: PathBuf,
    #[command(subcommand)]
    pub command: Command,
}

#[derive(Subcommand)]
pub enum Command {
    Tokenize {
        #[arg(short, long, help = "number of tokens in vocabulary generated")]
        num_tokens: usize,
        #[arg(short, long, help = "text file with corpus")]
        corpus: PathBuf,
        #[arg(long, help = "merge characters instead of bytes")]
        chars: bool,
    },
    Train {
        #[arg(short, long, help = "model parameters file")]
        model: PathBuf,
        #[arg(short, long, default_value = "8", help = "number of epochs")]
        epochs: usize,
        #[arg(short, long, help = "Saved model to start with")]
        checkpoint: Option<PathBuf>,
        #[arg(
            long,
            help = "stop each epoch after N batches, skipping validation and checkpoints - for benchmarking"
        )]
        max_batches: Option<usize>,
    },
    /// Sample text from a trained model
    Sample {
        #[arg(short, long, help = "model parameters file")]
        model: PathBuf,
        #[arg(short, long, help = "Context to start with")]
        context: String,
        #[arg(short, long, help = "Number of tokens to generate")]
        size: usize,
        // -t is taken by the global --tokenizer
        #[arg(short = 'T', long, default_value = "1.0", help = "Temperature")]
        temp: f64,
        #[arg(long, help = "top-k to sample from token probabilities")]
        top_k: Option<usize>,
        #[arg(long, help = "top-p to sample from token probabilities")]
        top_p: Option<f32>,
    },
}
