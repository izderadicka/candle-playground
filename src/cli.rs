use clap::{Parser, Subcommand};
use std::path::PathBuf;

#[derive(Parser)]
pub struct Cli {
    /// Model file
    #[arg(short, long, help = "model parameters file")]
    pub file: PathBuf,
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
        #[arg(short, long, default_value = "32", help = "number of epochs")]
        epochs: usize,
        #[arg(short, long, help = "Saved model to start with")]
        checkpoint: Option<PathBuf>,
    },
    /// Sample text from a trained model
    Sample {
        #[arg(short, long, help = "Context to start with")]
        context: String,
        #[arg(short, long, help = "Number of characters to generate")]
        size: usize,
        #[arg(short, long, default_value = "1.0", help = "Temperature")]
        temp: f64,
        #[arg(long, help = "top-k to sample from token probabilities")]
        top_k: Option<usize>,
        #[arg(long, help = "top-p to sample from token probabilities")]
        top_p: Option<f32>,
    },
}
