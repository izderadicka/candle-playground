use std::{collections::HashMap, path::Path};

use anyhow::Result;
use candle_core::{Device, Tensor};
use rand::{RngExt, SeedableRng, rngs::StdRng, seq::SliceRandom};

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

fn byte_to_vector(byte: u8, vocab_size: usize) -> Vec<f32> {
    let mut vec = vec![0.0; vocab_size];
    vec[byte as usize] = 1.0;
    vec
}

fn char_to_vector(c: char, char_to_index: &HashMap<char, u8>, vocab_size: usize) -> Vec<f32> {
    let index = char_to_index.get(&c).unwrap();
    byte_to_vector(*index, vocab_size)
}

fn text_to_vectors(
    text: &str,
    char_to_index: &HashMap<char, u8>,
    vocab_size: usize,
) -> Vec<Vec<f32>> {
    text.chars()
        .map(|c| char_to_vector(c, char_to_index, vocab_size))
        .collect()
}

fn top_k_chars(char_counts: &HashMap<char, usize>, k: usize) -> Vec<(char, usize)> {
    let mut char_counts_vec: Vec<(char, usize)> =
        char_counts.iter().map(|(&c, &count)| (c, count)).collect();
    char_counts_vec.sort_by(|a, b| b.1.cmp(&a.1));
    char_counts_vec.truncate(k);
    char_counts_vec
}

fn bottom_k_chars(char_counts: &HashMap<char, usize>, k: usize) -> Vec<(char, usize)> {
    let mut char_counts_vec: Vec<(char, usize)> =
        char_counts.iter().map(|(&c, &count)| (c, count)).collect();
    char_counts_vec.sort_by(|a, b| a.1.cmp(&b.1));
    char_counts_vec.truncate(k);
    char_counts_vec
}

const WINDOW_SIZE: usize = 100;

fn generate_batches(
    indices: &[u8],
    window_size: usize,
    batch_size: usize,
    rng: &mut StdRng,
) -> Result<Vec<Vec<Vec<u8>>>> {
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

fn window_to_input(window: &[u8], vocab_size: usize) -> Result<(Vec<Vec<f32>>, u8)> {
    let input = window[..window.len() - 1]
        .into_iter()
        .map(|&b| byte_to_vector(b, vocab_size))
        .collect::<Vec<_>>();
    let target = *window.last().unwrap();
    Ok((input, target))
}

fn batch_data(batch: Vec<Vec<u8>>, vocab_size: usize, dev: &Device) -> Result<(Tensor, Tensor)> {
    let batch_size = batch.len();
    let mut inputs = Vec::with_capacity(batch_size);
    let mut targets = Vec::with_capacity(batch_size);

    for window in batch {
        let (input, target) = window_to_input(&window, vocab_size)?;
        inputs.push(input);
        targets.push(target);
    }
    let input_tensor = Tensor::from_vec(inputs, (), dev)?;
    let target_tensor = Tensor::from_vec(targets, (batch_size,), dev)?;
    Ok((input_tensor, target_tensor))
}

fn main() -> anyhow::Result<()> {
    let corpus = read_text(Path::new("data/capek.txt"))?;
    let char_counts = collect_chars(&corpus);
    print!("Corpus length: {}\n", corpus.len());
    print!("Unique characters: {}\n", char_counts.len());
    let chars_map = chars_to_indices(&char_counts);
    let indices_map = indices_to_chars(&char_counts);
    let top_k = top_k_chars(&char_counts, 10);
    let corpus = corpus
        .chars()
        .map(|c| chars_map.get(&c).unwrap().to_owned())
        .collect::<Vec<u8>>();
    let mut rng = StdRng::seed_from_u64(42);

    Ok(())
}
