// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use serde_json::{Value, json};
use tempfile::tempdir;
use tokenizers::pre_tokenizers::byte_level::ByteLevel;

use super::{CacheConfig, CacheMode, EncodeCache, MIB};
use crate::{HuggingFaceTokenizer, Tokenizer};

const SPECIAL: &str = "<|s|>";
const SHIELD: &str = "x<|s|>x";
const SHARED_PREFIX: &str = "context: Café 日本語\n  ";

fn fixture() -> Value {
    let mut alphabet = ByteLevel::alphabet().into_iter().collect::<Vec<_>>();
    alphabet.sort_unstable();
    let mut vocab = alphabet
        .into_iter()
        .enumerate()
        .map(|(id, token)| (token.to_string(), json!(id)))
        .collect::<serde_json::Map<_, _>>();
    vocab.insert("ab".into(), json!(256));
    json!({
        "version": "1.0",
        "normalizer": null,
        "pre_tokenizer": {"type":"Sequence", "pretokenizers":[
            {"type":"Split", "pattern":{"Regex":"[a-z]+|[^a-z]+"}, "behavior":"Isolated", "invert":false},
            {"type":"ByteLevel", "add_prefix_space":false, "trim_offsets":true, "use_regex":false}
        ]},
        "post_processor": {"type":"TemplateProcessing",
            "single":[{"SpecialToken":{"id":SPECIAL,"type_id":0}}, {"Sequence":{"id":"A","type_id":0}}],
            "pair":[{"Sequence":{"id":"A","type_id":0}},{"Sequence":{"id":"B","type_id":1}}],
            "special_tokens":{(SPECIAL):{"id":SPECIAL,"ids":[257],"tokens":[SPECIAL]}}
        },
        "added_tokens":[
            {"id":257,"content":SPECIAL,"single_word":false,"lstrip":false,"rstrip":false,"normalized":false,"special":true},
            {"id":258,"content":SHIELD,"single_word":false,"lstrip":false,"rstrip":false,"normalized":false,"special":false}
        ],
        "model":{"type":"BPE","dropout":null,"vocab":vocab,"merges":[["a","b"]]}
    })
}

fn write_fixture(dir: &Path, value: &Value) -> PathBuf {
    let path = dir.join("tokenizer.json");
    fs::write(&path, serde_json::to_vec(value).unwrap()).unwrap();
    path
}

#[test]
fn encode_cache_matches_native_and_reuses_segments() {
    let dir = tempdir().unwrap();
    let path = write_fixture(dir.path(), &fixture());
    let native = HuggingFaceTokenizer::new_fastokens_with_cache(&path, None).unwrap();
    let hf = HuggingFaceTokenizer::new_hf_with_cache(&path, None).unwrap();
    let cached = HuggingFaceTokenizer::new_fastokens_with_cache(
        &path,
        Some(CacheConfig {
            l1_bytes: MIB,
            shared: None,
            epoch_file: None,
        }),
    )
    .unwrap();

    let pieces = [
        "",
        "abab",
        "\r\n hello",
        "Cafe\u{301} 🇺🇳",
        "日本語",
        SPECIAL,
        SHIELD,
    ];
    for a in pieces {
        for b in pieces {
            for c in pieces {
                let text = format!("{a}{b}{c}");
                assert_eq!(
                    cached.encode(&text, false).unwrap(),
                    native.encode(&text, false).unwrap(),
                    "{text:?}"
                );
                assert_eq!(
                    cached.encode(&text, false).unwrap(),
                    hf.encode(&text, false).unwrap(),
                    "{text:?}"
                );
            }
        }
    }
    assert_eq!(cached.encode(SHIELD, false).unwrap(), vec![258]);
    let first = format!("{SHARED_PREFIX}{SPECIAL}first suffix");
    let second = format!("{SHARED_PREFIX}{SPECIAL}other suffix");
    cached.encode(&first, false).unwrap();
    let before = cached.encode_cache_stats().unwrap();
    assert_eq!(
        cached.encode(&second, false).unwrap(),
        native.encode(&second, false).unwrap()
    );
    let after = cached.encode_cache_stats().unwrap();
    assert_eq!(after.reused_tokens - before.reused_tokens, native.encode(SHARED_PREFIX, false).unwrap().len() as u64);
    assert_eq!(after.encoded_tokens - before.encoded_tokens, native.encode("other suffix", false).unwrap().len() as u64);
    assert_eq!(after.special_tokens - before.special_tokens, 1);
    assert_eq!(after.l1_hits - before.l1_hits, 1);
    assert_eq!(after.misses - before.misses, 1);

    assert_eq!(
        cached.encode(&second, true).unwrap(),
        native.encode(&second, true).unwrap()
    );
    assert_eq!(
        cached.encode_ordinary(SPECIAL).unwrap(),
        native.encode_ordinary(SPECIAL).unwrap()
    );
    assert_eq!(cached.encode_cache_stats().unwrap(), after);
    assert!(native.encode_cache_stats().is_none());
}

#[test]
fn complex_added_tokens_use_whole_input_cache() {
    let dir = tempdir().unwrap();
    let mut config = fixture();
    config["added_tokens"][0]["lstrip"] = json!(true);
    let path = write_fixture(dir.path(), &config);
    let native = HuggingFaceTokenizer::new_fastokens_with_cache(&path, None).unwrap();
    let opted_in = HuggingFaceTokenizer::new_fastokens_with_cache(
        &path,
        Some(CacheConfig {
            l1_bytes: MIB,
            shared: None,
            epoch_file: None,
        }),
    )
    .unwrap();
    assert_eq!(opted_in.encode_cache_stats().unwrap().l1_hits, 0);
    assert_eq!(
        opted_in.encode("ab  <|s|>ab", false).unwrap(),
        native.encode("ab  <|s|>ab", false).unwrap()
    );

    let mut config = fixture();
    config["added_tokens"][1]["normalized"] = json!(true);
    let path = write_fixture(dir.path(), &config);
    let opted_in = HuggingFaceTokenizer::new_fastokens_with_cache(
        &path,
        Some(CacheConfig {
            l1_bytes: MIB,
            shared: None,
            epoch_file: None,
        }),
    )
    .unwrap();
    let prompt = format!("ab{SPECIAL}cd");
    let before = opted_in.encode_cache_stats().unwrap();
    opted_in.encode(&prompt, false).unwrap();
    let after = opted_in.encode_cache_stats().unwrap();
    assert_eq!(
        after.misses - before.misses,
        1,
        "whole input, not two segments"
    );
    opted_in.encode(&prompt, false).unwrap();
    assert_eq!(opted_in.encode_cache_stats().unwrap().l1_hits, 1);
    let hf_native = HuggingFaceTokenizer::new_hf_with_cache(&path, None).unwrap();
    let hf_cached = HuggingFaceTokenizer::new_hf_with_cache(
        &path,
        Some(CacheConfig {
            l1_bytes: MIB,
            shared: None,
            epoch_file: None,
        }),
    )
    .unwrap();
    assert_eq!(
        hf_cached.encode(SHIELD, false).unwrap(),
        hf_native.encode(SHIELD, false).unwrap()
    );
    assert_eq!(
        hf_cached.encode(SHIELD, false).unwrap(),
        hf_native.encode(SHIELD, false).unwrap()
    );
    assert_eq!(hf_cached.encode_cache_stats().unwrap().l1_hits, 1);
}

#[test]
fn l1_evicts_oldest_and_bypasses_oversize_segments() {
    let cache = EncodeCache::new(600, None, None, CacheMode::FastokensSegments);
    let get = |text, expected| {
        let mut ids = Vec::new();
        cache
            .append_segment("", None, text, &mut ids, || Ok::<_, ()>(vec![expected]))
            .unwrap();
        ids
    };
    assert_eq!(get("old", 1), [1]);
    assert_eq!(get("new", 2), [2]);
    assert_eq!(get("old", 3), [1]);
    assert_eq!(get("third", 4), [4]);
    assert_eq!(get("new", 5), [5]);
    let before = cache.stats();
    assert!(before.l1_bytes <= 600);
    assert_eq!(before.l1_hits, 1);
    assert_eq!(get(&"x".repeat(1000), 6), [6]);
    assert_eq!(cache.stats().l1_bytes, before.l1_bytes);
}

#[test]
fn shared_cache_cross_process() {
    const CHILD: &str = "VLLM_RS_CACHE_TEST_CHILD";
    const PATH: &str = "VLLM_RS_CACHE_TEST_MODEL";
    if let Some(mode) = std::env::var_os(CHILD) {
        let path = PathBuf::from(std::env::var_os(PATH).unwrap());
        // Exercise the serving constructor and the trait's encode method.
        let tokenizer = HuggingFaceTokenizer::new(&path).unwrap();
        let native = HuggingFaceTokenizer::new_hf_with_cache(&path, None).unwrap();
        let prompt = format!("{SHARED_PREFIX}{SPECIAL}");
        assert_eq!(
            tokenizer.encode(&prompt, false).unwrap(),
            native.encode(&prompt, false).unwrap()
        );
        let stats = tokenizer.encode_cache_stats();
        if mode == "disabled" {
            assert!(stats.is_none());
        } else {
            let stats = stats.unwrap();
            assert_eq!(stats.l1_hits, 0);
            assert_eq!(stats.shared_hits, u64::from(mode == "hit"));
            assert_eq!(stats.misses, u64::from(mode != "hit"));
            assert_eq!(CacheConfig::from_env().unwrap().l1_bytes, 128 * MIB);
            if mode == "mismatch" {
                // Only sharing is disabled for an incompatible model.
                assert!(tokenizer.encode_cache.as_ref().unwrap().shared.is_none());
                assert_eq!(
                    tokenizer.encode(&prompt, false).unwrap(),
                    native.encode(&prompt, false).unwrap()
                );
                assert_eq!(tokenizer.encode_cache_stats().unwrap().l1_hits, 1);
            }
        }
        return;
    }

    let dir = tempdir().unwrap();
    let path = write_fixture(dir.path(), &fixture());
    let db = dir.path().join("segments.db");
    fs::write(&db, b"python pool remains separate").unwrap();
    let pool = dir.path().join("segments.db.rust-021-v1");
    let epoch = dir.path().join("epoch");
    fs::write(&epoch, "generation-a\n").unwrap();
    let tokenizer = HuggingFaceTokenizer::new_fastokens_with_cache(
        &path,
        Some(CacheConfig {
            l1_bytes: MIB,
            shared: Some((pool.clone(), MIB, "glm53".into())),
            epoch_file: Some(epoch.clone()),
        }),
    )
    .unwrap();
    let prefix = format!("{SHARED_PREFIX}{SPECIAL}");
    tokenizer.encode(&prefix, false).unwrap();
    assert_eq!(tokenizer.encode_cache_stats().unwrap().misses, 1);

    let child = |mode, namespace, model: &Path| {
        let mut command = Command::new(std::env::current_exe().unwrap());
        command
            .args([
                "--exact",
                "hf::encode_cache::tests::shared_cache_cross_process",
                "--nocapture",
            ])
            .env(CHILD, mode)
            .env(PATH, model)
            .env_remove("FASTOKENS_SEGMENT_CACHE")
            .env_remove("VLLM_RS_TOKENIZER_NAMESPACE")
            .env("VLLM_GLM53_TOKENIZER_CACHE_GIB", "0")
            .env("VLLM_GLM53_TOKENIZER_SHARED_PATH", &db)
            .env("VLLM_GLM53_TOKENIZER_SHARED_GIB", "0.0009765625")
            .env("VLLM_GLM53_TOKENIZER_LOCAL_L1_MIB", "128")
            .env("VLLM_GLM53_TOKENIZER_CACHE_EPOCH_FILE", &epoch);
        if mode != "disabled" {
            command.env("VLLM_GLM53_TOKENIZER_CACHE_GIB", "2");
            if let Some(namespace) = namespace {
                command.env("VLLM_RS_TOKENIZER_NAMESPACE", namespace);
            }
        }
        let output = command.output().unwrap();
        assert!(
            output.status.success(),
            "{mode}: {}\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        assert!(String::from_utf8_lossy(&output.stdout).contains("1 passed"));
    };
    child("hit", None, &path);
    child("miss", Some("tenant-b"), &path);
    child("disabled", None, &path);
    let pending = dir.path().join("epoch.new");
    fs::write(&pending, "generation-b\n").unwrap();
    fs::rename(&pending, &epoch).unwrap();
    child("miss", None, &path);
    // The original process is still alive, and its old L1 must not win.
    tokenizer.encode(&prefix, false).unwrap();
    let stats = tokenizer.encode_cache_stats().unwrap();
    assert_eq!(stats.l1_hits, 0);
    assert_eq!(stats.shared_hits, 1);
    assert_eq!(fs::read(&db).unwrap(), b"python pool remains separate");
    fs::remove_file(&epoch).unwrap();
    tokenizer.encode(&prefix, false).unwrap();
    assert_eq!(
        tokenizer.encode_cache_stats().unwrap(),
        crate::EncodeCacheStats {
            l1_bytes: 0,
            ..stats
        }
    );
    fs::write(&epoch, "generation-b\n").unwrap();

    let changed = dir.path().join("changed");
    fs::create_dir(&changed).unwrap();
    let changed_path = write_fixture(&changed, &fixture());
    fs::write(changed.join("tokenizer_config.json"), br#"{"added_tokens_decoder":{"259":{"content":"<|new|>","normalized":false,"special":true}}}"#).unwrap();
    child("mismatch", None, &changed_path);
}

#[test]
fn shared_slots_bound_all_epochs_and_reject_corruption() {
    use super::shared::SharedPool;
    let dir = tempdir().unwrap();
    let path = dir.path().join("pool");
    let pool = SharedPool::open(path.clone(), MIB, "glm53".into(), &fixture()).unwrap();
    let input = "request contents should not be written";
    for generation in 0..1024 {
        let key = pool.key(&generation.to_string(), 0, input).unwrap();
        pool.put(&key, &[7; 100]);
    }
    let mut slots = 0;
    let mut bytes = 0;
    for entry in fs::read_dir(&path).unwrap().map(Result::unwrap) {
        if entry.file_name().to_string_lossy().starts_with("slot-") {
            slots += 1;
            let record = fs::read(entry.path()).unwrap();
            assert!(!record.windows(input.len()).any(|part| part == input.as_bytes()));
            bytes += record.len();
        }
    }
    assert!(slots <= 256 && bytes <= MIB);
    let key = pool.key("1023", 0, input).unwrap();
    assert_eq!(pool.get(&key).unwrap(), [7; 100]);
    let slot = pool.slot_path(&key);
    let mut record = fs::read(&slot).unwrap();
    record[32] ^= 1;
    fs::write(slot, record).unwrap();
    assert!(pool.get(&key).is_none());
}

#[test]
#[ignore = "requires the deployed tokenizer.json and its sibling tokenizer_config.json"]
fn deployed_glm53_cache_parity() {
    let path = PathBuf::from(
        std::env::var_os("VLLM_TEST_GLM53_TOKENIZER_PATH")
            .expect("set VLLM_TEST_GLM53_TOKENIZER_PATH to the deployed tokenizer.json"),
    );
    assert!(path.with_file_name("tokenizer_config.json").is_file());
    let hf = HuggingFaceTokenizer::new_hf_with_cache(&path, None).unwrap();
    let dir = tempdir().unwrap();
    let config = || {
        Some(CacheConfig {
            l1_bytes: MIB,
            shared: Some((dir.path().join("pool"), 16 * MIB, "glm53".into())),
            epoch_file: None,
        })
    };
    let fast = HuggingFaceTokenizer::new_fastokens_with_cache(&path, None).ok();
    let native = fast.as_ref().unwrap_or(&hf);
    let load = |config| {
        if fast.is_some() {
            HuggingFaceTokenizer::new_fastokens_with_cache(&path, config)
        } else {
            HuggingFaceTokenizer::new_hf_with_cache(&path, config)
        }
        .unwrap()
    };
    let cached = load(config());

    for prompt in [
        "",
        "你好，世界！ café \r\n  ",
        "[gMASK]<sop><|user|>payload<|assistant|>",
        "<|system|>system\n<|user|>diff --git a/x b/x\n<|assistant|>",
    ] {
        let expected = native.encode(prompt, false).unwrap();
        assert_eq!(cached.encode(prompt, false).unwrap(), expected);
        assert_eq!(
            cached.encode(prompt, false).unwrap(),
            hf.encode(prompt, false).unwrap()
        );
    }
    // Repeat the last prompt explicitly: whole-input mode cannot get L1 hits
    // merely by seeing the same prefix in different prompts.
    let prompt = "<|system|>system\n<|user|>diff --git a/x b/x\n<|assistant|>";
    let expected = hf.encode(prompt, false).unwrap();
    assert_eq!(cached.encode(prompt, false).unwrap(), expected);
    let stats = cached.encode_cache_stats().expect("deployment tokenizer did not enable cache");
    assert!(stats.l1_hits > 0 && stats.misses > 0, "{stats:?}");
    let fresh = load(config());
    assert_eq!(fresh.encode(prompt, false).unwrap(), expected);
    let fresh_stats = fresh.encode_cache_stats().unwrap();
    assert!(fresh_stats.shared_hits > 0, "{fresh_stats:?}");
    let mode = cached.encode_cache.as_ref().unwrap().mode;
    eprintln!("GLM-5.3 mode={mode:?}, original={stats:?}, fresh={fresh_stats:?}");
}

#[test]
fn request_salt_separates_both_cache_tiers() {
    let dir = tempdir().unwrap();
    let path = write_fixture(dir.path(), &fixture());
    let new = || HuggingFaceTokenizer::new_fastokens_with_cache(&path, Some(CacheConfig {
        l1_bytes: MIB, shared: Some((dir.path().join("pool"), MIB, "test".into())), epoch_file: None,
    })).unwrap();
    let a = new();
    let prompt = "shared text";
    let first = a.encode_scoped(prompt, false, Some("tenantA")).unwrap();
    assert_eq!(a.encode_scoped(prompt, false, Some("tenantB")).unwrap(), first);
    assert_eq!(a.encode_cache_stats().unwrap().misses, 2);
    let b = new();
    assert_eq!(b.encode_scoped(prompt, false, Some("tenantB")).unwrap(), first);
    assert_eq!(b.encode_cache_stats().unwrap().shared_hits, 1);
    b.encode_scoped(prompt, false, None).unwrap();
    b.encode_scoped("\0s7:tenantAshared text", false, None).unwrap();
    assert_eq!(b.encode_cache_stats().unwrap().misses, 2);
}
