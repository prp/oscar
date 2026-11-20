//! The `kimi-k2` pre-tokenizer split K3 loads through, under the `tokenizers` regex backend.

use mistralrs_core::gguf::{convert_gguf_to_hf_tokenizer, Content, SPLIT_GPT2, SPLIT_KIMI_K2};
use tokenizers::pre_tokenizers::split::{Split, SplitPattern};
use tokenizers::tokenizer::normalizer::SplitDelimiterBehavior;
use tokenizers::{OffsetReferential, OffsetType, PreTokenizedString, PreTokenizer};

fn pieces_with(pattern: &str, s: &str) -> Vec<String> {
    let split = Split::new(SplitPattern::Regex(pattern.to_string()), SplitDelimiterBehavior::Isolated, false)
        .expect("split regex must compile");
    let mut pre = PreTokenizedString::from(s);
    split.pre_tokenize(&mut pre).expect("pre_tokenize");
    pre.get_splits(OffsetReferential::Original, OffsetType::Byte).into_iter().map(|(t, _, _)| t.to_string()).collect()
}

#[test]
fn applies_its_three_distinguishing_rules() {
    let k2 = |s: &str| pieces_with(SPLIT_KIMI_K2, s);
    let gpt2 = |s: &str| pieces_with(SPLIT_GPT2, s);

    assert_eq!(k2("1234567"), vec!["123", "456", "7"]);
    assert!(k2("abc汉字def").contains(&"汉字".to_string()));
    // Plain prose splits as with `SPLIT_GPT2`; a contraction does not, K2 attaching `'s` as a suffix.
    assert_eq!(k2("The Eiffel Tower"), gpt2("The Eiffel Tower"));
    assert_ne!(k2("Mercator's"), gpt2("Mercator's"));
}

/// The oracle ids are llama.cpp's hand-coded K2 splitter's; needs `K3_GGUF_DIR`.
#[test]
#[ignore]
fn matches_llama_cpp_on_a_real_gguf() -> anyhow::Result<()> {
    let dir = std::env::var("K3_GGUF_DIR")
        .map_err(|_| anyhow::anyhow!("K3_GGUF_DIR names the directory of K3's UD-IQ1_M shards"))?;
    let mut files: Vec<std::fs::File> = (1..=15)
        .map(|i| std::fs::File::open(format!("{dir}/Kimi-K3-UD-IQ1_M-{i:05}-of-00015.gguf")))
        .collect::<std::io::Result<_>>()?;
    let mut readers: Vec<&mut std::fs::File> = files.iter_mut().collect();
    let content = Content::from_readers(&mut readers)?;
    let conv = convert_gguf_to_hf_tokenizer(&content)?;

    for (prompt, oracle) in [
        (
            "Revenue rose 1234567 units in 2026.",
            vec![902u32, 11003, 23220, 220, 6694, 12972, 22, 11591, 306, 220, 2975, 21, 13],
        ),
        ("北京的天气很好today", vec![42930u32, 8597, 6523, 86167]),
        (
            "Mercator's projection and the hiker's map—don't PANIC.",
            vec![135346u32, 1500, 1048, 37022, 316, 276, 324, 36691, 1048, 4421, 3442, 88709, 113854, 1883, 13],
        ),
    ] {
        let ids = conv.tokenizer.encode_fast(prompt, false).map_err(anyhow::Error::msg)?.get_ids().to_vec();
        assert_eq!(ids, oracle, "kimi-k2 split diverges from llama.cpp on {prompt:?}");
    }
    Ok(())
}
