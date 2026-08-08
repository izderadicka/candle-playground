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
    /// Train the model and save it to the model file
    Train {
        #[arg(short, long, default_value = "32", help = "number of epochs")]
        epochs: usize,
    },
    /// Sample text from a trained model
    Sample {
        #[arg(short, long, help = "Context to start with")]
        context: String,
        #[arg(short, long, help = "Number of characters to generate")]
        size: usize,
        #[arg(short, long, default_value = "1.0", help = "Temperature")]
        temp: f64,
    },
}
