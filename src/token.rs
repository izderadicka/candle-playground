use candle_core::{Device, Tensor};
use rand::{RngExt, seq::SliceRandom};
use tokenizers::models::bpe::{BPE, BpeTrainerBuilder};
use tokenizers::normalizers::{strip::Strip, unicode::NFC, utils::Sequence};
use tokenizers::pre_tokenizers::byte_level::ByteLevel;
use tokenizers::pre_tokenizers::metaspace::{Metaspace, PrependScheme};
use tokenizers::processors::PostProcessorWrapper;
use tokenizers::{AddedToken, Result, Tokenizer, TokenizerBuilder};

use std::ffi::OsStr;
use std::path::Path;

pub const UNK: &str = "<unk>";

pub fn train(corpus: impl AsRef<Path>, vocab_size: usize, output: impl AsRef<Path>) -> Result<()> {
    let corpus: &OsStr = corpus.as_ref().as_ref();
    let corpus: String = corpus.to_string_lossy().into();
    let mut trainer = BpeTrainerBuilder::new()
        .show_progress(true)
        .vocab_size(vocab_size)
        .min_frequency(0)
        // .special_tokens(vec![
        //     AddedToken::from(String::from("<s>"), true),
        //     AddedToken::from(String::from("<pad>"), true),
        //     AddedToken::from(String::from("</s>"), true),
        //     AddedToken::from(String::from("<unk>"), true),
        //     AddedToken::from(String::from("<mask>"), true),
        // ])
        .build();

    let mut tokenizer = TokenizerBuilder::new()
        .with_model(BPE::default())
        .with_normalizer(Some(Sequence::new(vec![
            Strip::new(true, true).into(),
            NFC.into(),
        ])))
        .with_pre_tokenizer(Some(ByteLevel::default()))
        .with_post_processor(Some(ByteLevel::default()))
        .with_decoder(Some(ByteLevel::default()))
        .build()?;

    let pretty = true;
    tokenizer
        .train_from_files(&mut trainer, vec![corpus])?
        .save(output, pretty)?;

    Ok(())
}

/// Same BPE as [`train`] - tunable `vocab_size`, real merges - but the base
/// alphabet is unicode characters instead of bytes.
///
/// The alphabet is decided by the pre-tokenizer: `ByteLevel` maps the text to
/// its 256 bytes first, `Metaspace` leaves the characters alone (it only turns
/// spaces into `▁` so that word boundaries survive as ordinary characters).
/// The trainer then seeds the vocabulary with every character it sees and
/// merges from there.
pub fn train_chars(
    corpus: impl AsRef<Path>,
    vocab_size: usize,
    output: impl AsRef<Path>,
) -> Result<()> {
    let corpus: &OsStr = corpus.as_ref().as_ref();
    let corpus: String = corpus.to_string_lossy().into();

    let mut trainer = BpeTrainerBuilder::new()
        .show_progress(true)
        .vocab_size(vocab_size)
        .min_frequency(0)
        // characters are the base alphabet, so an unseen one at inference time
        // has no byte fallback - it needs an <unk>
        .special_tokens(vec![AddedToken::from(String::from(UNK), true)])
        // every character of the corpus enters the alphabet by default - add
        // .limit_alphabet(200) to cap the number of single-char tokens
        .build();

    // '▁' + Always: a leading marker on every word, so "ahoj" and " ahoj"
    // tokenize the same way and decoding can restore the spaces
    let metaspace = Metaspace::new('▁', PrependScheme::Always, true);

    let mut tokenizer = TokenizerBuilder::new()
        .with_model(BPE::builder().unk_token(String::from(UNK)).build()?)
        .with_normalizer(Some(NFC))
        .with_pre_tokenizer(Some(metaspace.clone()))
        .with_post_processor(None::<PostProcessorWrapper>)
        .with_decoder(Some(metaspace))
        .build()?;

    let pretty = true;
    tokenizer
        .train_from_files(&mut trainer, vec![corpus])?
        .save(output, pretty)?;

    Ok(())
}

/// Text -> token ids.
///
/// A free function rather than a [`Corpus`] method because the input differs:
/// training encodes a whole corpus file, sampling encodes a short seed string.
pub fn encode(tokenizer: &Tokenizer, text: &str) -> anyhow::Result<Vec<u32>> {
    let encoding = tokenizer
        // no post processor, so there are no special tokens to add
        .encode(text, false)
        .map_err(|e| anyhow::anyhow!(e))?;
    Ok(encoding.get_ids().to_vec())
}

/// A tokenizer plus, optionally, a corpus already turned into token ids.
///
/// Sampling only needs the tokenizer, training needs both - hence the two
/// constructors.
pub struct Corpus {
    tokenizer: Tokenizer,
    tokens: Vec<u32>,
}

impl Corpus {
    /// Just the tokenizer - all that sampling needs.
    pub fn from_tokenizer_file(tokenizer: impl AsRef<Path>) -> anyhow::Result<Self> {
        let tokenizer = Tokenizer::from_file(tokenizer).map_err(|e| anyhow::anyhow!(e))?;
        Ok(Self {
            tokenizer,
            tokens: Vec::new(),
        })
    }

    /// Tokenizer plus a text file encoded with it - what training needs.
    pub fn from_files(
        tokenizer: impl AsRef<Path>,
        text_file: impl AsRef<Path>,
    ) -> anyhow::Result<Self> {
        let mut corpus = Self::from_tokenizer_file(tokenizer)?;
        let text = std::fs::read_to_string(text_file.as_ref())?;
        corpus.tokens = encode(&corpus.tokenizer, &text)?;

        println!("Corpus length: {} chars", text.chars().count());
        println!("Corpus tokens: {}", corpus.tokens.len());
        println!("Vocabulary size: {}", corpus.vocab_size());

        Ok(corpus)
    }

    pub fn tokenizer(&self) -> &Tokenizer {
        &self.tokenizer
    }

    pub fn tokens(&self) -> &[u32] {
        &self.tokens
    }

    pub fn vocab_size(&self) -> usize {
        self.tokenizer.get_vocab_size(true)
    }

    pub fn encode(&self, text: &str) -> anyhow::Result<Vec<u32>> {
        encode(&self.tokenizer, text)
    }

    pub fn decode(&self, ids: &[u32]) -> anyhow::Result<String> {
        self.tokenizer
            .decode(ids, false)
            .map_err(|e| anyhow::anyhow!(e))
    }
}

pub type Batches = Vec<Vec<Vec<u32>>>;

/// Same as [`crate::text::generate_batches`], on token ids instead of char indices.
pub fn generate_batches(
    tokens: &[u32],
    window_size: usize,
    batch_size: usize,
    rng: &mut impl RngExt,
) -> anyhow::Result<Batches> {
    // `tokens.len() - window_size` below wraps rather than panicking in release
    if tokens.len() <= window_size {
        anyhow::bail!(
            "not enough tokens to batch: {} tokens for a window of {window_size}",
            tokens.len()
        );
    }
    let start = rng.random_range(0..window_size);
    let mut window_starts: Vec<usize> = (start..tokens.len() - window_size)
        .step_by(window_size)
        .collect();
    window_starts.shuffle(rng);
    let windows = window_starts
        .into_iter()
        .map(|start| &tokens[start..start + window_size + 1])
        .map(Vec::from)
        .collect::<Vec<_>>();
    let batches = windows
        .chunks(batch_size)
        .map(Vec::from)
        .collect::<Vec<_>>();

    Ok(batches)
}

/// Same as [`crate::text::batch_data`], on token ids instead of char indices.
///
/// Targets are the inputs shifted one position left, flattened row major to
/// match the `(batch * seq, vocab)` reshape the loss expects.
pub fn batch_data(batch: &[Vec<u32>], dev: &Device) -> anyhow::Result<(Tensor, Tensor)> {
    let batch_size = batch.len();
    let seq_len = batch[0].len() - 1;
    let mut inputs = Vec::with_capacity(batch_size * seq_len);
    let mut targets = Vec::with_capacity(batch_size * seq_len);

    for window in batch.iter() {
        inputs.extend_from_slice(&window[..seq_len]);
        targets.extend_from_slice(&window[1..]);
    }

    let input_tensor = Tensor::from_vec(inputs, (batch_size, seq_len), dev)?;
    let target_tensor = Tensor::from_vec(targets, (batch_size * seq_len,), dev)?;
    Ok((input_tensor, target_tensor))
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOKENIZER: &str = "data/capek-tokens.json";
    const CORPUS: &str = "data/capek.txt";

    /// Encoding and decoding must be lossless, newlines and tabs included -
    /// the model is trained on the ids, but we read the decoded text.
    #[test]
    fn round_trip_on_real_text() -> anyhow::Result<()> {
        let corpus = Corpus::from_tokenizer_file(TOKENIZER)?;
        let text: String = std::fs::read_to_string(CORPUS)?.chars().take(2000).collect();

        let ids = corpus.encode(&text)?;
        assert_eq!(corpus.decode(&ids)?, text);
        Ok(())
    }

    #[test]
    fn every_id_is_usable() -> anyhow::Result<()> {
        let corpus = Corpus::from_files(TOKENIZER, CORPUS)?;
        let vocab_size = corpus.vocab_size();

        assert!(!corpus.tokens().is_empty());
        // an id past the vocabulary would index past the embedding table
        for id in corpus.tokens() {
            assert!((*id as usize) < vocab_size, "id {id} >= vocab {vocab_size}");
        }
        // the model can emit any id in 0..vocab_size while sampling, and
        // Tokenizer::decode drops ids it cannot resolve instead of failing
        for id in 0..vocab_size as u32 {
            assert!(
                corpus.tokenizer().id_to_token(id).is_some(),
                "id {id} decodes to nothing"
            );
        }
        Ok(())
    }

    /// How well the corpus compresses at each vocabulary size, which is what
    /// decides the cost of an epoch: a bigger vocabulary means a bigger output
    /// layer but fewer tokens to train on.
    ///
    /// `cargo test --lib -- --ignored --nocapture`
    #[test]
    #[ignore = "trains several tokenizers, takes a minute"]
    fn compression_by_vocab_size() -> anyhow::Result<()> {
        let dir = std::env::temp_dir().join("capek-vocab-sweep");
        std::fs::create_dir_all(&dir)?;
        let chars = std::fs::read_to_string(CORPUS)?.chars().count();

        println!("\ncorpus: {chars} chars");
        println!("{:>6} {:>10} {:>12} {:>12}", "vocab", "tokens", "chars/token", "windows@200");
        for vocab_size in [1000, 2000, 4000] {
            let file = dir.join(format!("{vocab_size}.json"));
            train_chars(CORPUS, vocab_size, &file).map_err(|e| anyhow::anyhow!(e))?;

            let corpus = Corpus::from_files(&file, CORPUS)?;
            let tokens = corpus.tokens().len();
            println!(
                "{:>6} {:>10} {:>12.2} {:>12}",
                corpus.vocab_size(),
                tokens,
                chars as f64 / tokens as f64,
                tokens / 200
            );
        }
        Ok(())
    }

    /// On a ramp, every target must be its input plus one - that pins both the
    /// shift by one and the row major flattening the loss reshape relies on.
    #[test]
    fn batches_are_shifted_by_one() -> anyhow::Result<()> {
        let tokens: Vec<u32> = (0..1000).collect();
        let (window_size, batch_size) = (10, 8);
        let mut rng = rand::rng();

        let batches = generate_batches(&tokens, window_size, batch_size, &mut rng)?;
        assert!(!batches.is_empty());

        for batch in &batches {
            assert!(batch.len() <= batch_size);
            for window in batch {
                assert_eq!(window.len(), window_size + 1);
            }

            let (inputs, targets) = batch_data(batch, &Device::Cpu)?;
            assert_eq!(inputs.dims(), &[batch.len(), window_size]);
            assert_eq!(targets.dims(), &[batch.len() * window_size]);

            let inputs = inputs.flatten_all()?.to_vec1::<u32>()?;
            let targets = targets.to_vec1::<u32>()?;
            for (i, t) in inputs.iter().zip(targets.iter()) {
                assert_eq!(*t, i + 1);
            }
        }
        Ok(())
    }
}
