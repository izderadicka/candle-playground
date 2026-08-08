use std::{collections::HashMap, path::Path};

use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use rand::{RngExt, seq::SliceRandom};

pub type Batches = Vec<Vec<Vec<u8>>>;

pub fn generate_batches(
    indices: &[u8],
    window_size: usize,
    batch_size: usize,
    rng: &mut impl RngExt,
) -> Result<Batches> {
    let start = rng.random_range(0..window_size);
    let mut window_starts: Vec<usize> = (start..indices.len() - window_size)
        .step_by(window_size)
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

pub fn batch_data(batch: &[Vec<u8>], dev: &Device) -> Result<(Tensor, Tensor)> {
    let batch_size = batch.len();
    let seq_len = batch[0].len() - 1;
    let mut inputs = Vec::with_capacity(batch_size * seq_len);
    let mut targets = Vec::with_capacity(batch_size * seq_len);

    for window in batch.iter() {
        inputs.extend(window[..seq_len].iter().map(|x| *x as u32));
        targets.extend(window[1..].iter().map(|x| *x as u32));
    }

    let input_tensor = Tensor::from_vec(inputs, (batch_size, seq_len), dev)?;
    let target_tensor =
        Tensor::from_vec(targets, (batch_size * seq_len,), dev)?.to_dtype(DType::U32)?;
    Ok((input_tensor, target_tensor))
}

pub struct Corpus {
    vocab_size: usize,
    chars_map: HashMap<char, u8>,
    indices_map: HashMap<u8, char>,
    corpus: String,
}

impl Corpus {
    pub fn from_text_file(file: impl AsRef<Path>) -> Result<Self> {
        let corpus = std::fs::read_to_string(file.as_ref())?;
        let char_counts = collect_chars(&corpus);
        let vocab_size = char_counts.len();
        print!("Corpus length: {}\n", corpus.len());
        print!("Unique characters: {}\n", char_counts.len());
        let chars_map = chars_to_indices(&char_counts);
        let indices_map = indices_to_chars(&char_counts);

        Ok(Self {
            corpus,
            vocab_size,
            chars_map,
            indices_map,
        })
    }

    pub fn indexed_corpus(&self) -> Vec<u8> {
        self.corpus
            .chars()
            .map(|c| *self.chars_map.get(&c).unwrap()) // can unwrap as map is done from same data
            .collect()
    }

    pub fn char_to_index(&self, c: char) -> Result<u8> {
        self.chars_map
            .get(&c)
            .map(|i| *i)
            .ok_or_else(|| anyhow::anyhow!("Invalid char - not in corpus"))
    }

    pub fn index_to_char(&self, i: u8) -> Result<char> {
        self.indices_map
            .get(&i)
            .map(|c| *c)
            .ok_or_else(|| anyhow::anyhow!("Invalid index - not in corpus"))
    }

    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }
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
