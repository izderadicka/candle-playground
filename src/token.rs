use tokenizers::models::bpe::{BPE, BpeTrainerBuilder};
use tokenizers::normalizers::{strip::Strip, unicode::NFC, utils::Sequence};
use tokenizers::pre_tokenizers::byte_level::ByteLevel;
use tokenizers::pre_tokenizers::metaspace::{Metaspace, PrependScheme};
use tokenizers::processors::PostProcessorWrapper;
use tokenizers::{AddedToken, Result, TokenizerBuilder};

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
