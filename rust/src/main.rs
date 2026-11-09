//! Cross-validation harness: runs gliner25-rs (native Rust on ONNX Runtime)
//! over a windows JSONL file and writes predictions in the shared format used
//! by the CoreML and PyTorch runners in this repo.
//!
//! Schema (entity labels, relation names and the shared threshold) is read
//! from `schema/ontonotes.json` -- never hard-coded -- so the three engines
//! stay in lockstep when that file changes.
//!
//! Usage:
//!   ORT_DYLIB_PATH=/opt/homebrew/opt/onnxruntime/lib/libonnxruntime.dylib \
//!   gliner-rust-eval \
//!     --schema schema/ontonotes.json \
//!     --models-dir models/gliner2.5-multi-v1-onnx \
//!     --windows data/windows.jsonl \
//!     --out data/preds/rust-ort-fp32.jsonl \
//!     --device cpu --precision fp32
//!
//! ## Byte vs. char offsets (fixed 2026-11-09)
//!
//! `gliner25_rs::Mention::char_start`/`char_end` are, despite the name, BYTE
//! offsets into the `&str` -- see `boundary.rs:108` ("Byte range `[start,
//! end)` in the original text.") and `processor.rs:148` (same wording for
//! `word_to_char_maps`). The shared prediction format used across all three
//! engines in this repo is *char* (Unicode scalar value) offsets, matching
//! Python string indexing. On any window containing a multi-byte UTF-8
//! character (accented Latin letters in the NL/DE/FR gold sentences, but
//! also em dashes/arrows/section signs that show up constantly in the
//! "English" corpus text), the raw byte offset drifts from the char offset
//! by the number of extra bytes contributed by every multi-byte character at
//! or before that position -- silently misaligning every span after the
//! first non-ASCII character in a window. `byte_to_char_map` below converts
//! once per window before anything is written out.
//!
//! ## One-pass vs. two-pass
//!
//! The CoreML export's extraction bucket allows only 8 queries per call, so
//! `scripts/run_coreml.py` necessarily runs entities (8 queries) and
//! relations (4 relation names x 2 roles = 8 queries) as two separate
//! `extract()` calls per window. GLiNER2 conditions every task's attention
//! on the *whole* prompt, so a combined one-pass call is a materially
//! different input to the encoder than two one-task calls, not just a
//! latency difference. `--mode two-pass` (the default here, mirroring
//! CoreML) runs the same two separate calls; `--mode one-pass` runs the
//! original combined 12-query call, kept for the one-pass/two-pass cost
//! comparison in docs/cross-validation.md.

use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result};
use clap::Parser;
use serde::{Deserialize, Serialize};

use gliner25_rs::{BoundaryConfig, BoundaryEngine, BoundaryParams, SchemaTask, pair_relations};

#[derive(Parser, Debug)]
#[command(about = "Run gliner25-rs over a windows JSONL file, shared prediction format out")]
struct Args {
    /// schema/ontonotes.json -- single source of truth for labels & threshold
    #[arg(long)]
    schema: PathBuf,
    /// Directory holding the ONNX export (jugaadsrl/gliner2.5-multi-v1-onnx layout)
    #[arg(long)]
    models_dir: PathBuf,
    /// Windows JSONL ({window_id, chunk_id, char_start, char_end, text}) or gold.jsonl
    #[arg(long)]
    windows: PathBuf,
    /// Output predictions JSONL, shared format
    #[arg(long)]
    out: PathBuf,
    /// GLINER2_DEVICE: cpu | coreml | auto (see gliner25-rs runtime::execution_providers)
    #[arg(long, default_value = "cpu")]
    device: String,
    /// fp32 | fp16 | fp16_iobinding | auto
    #[arg(long, default_value = "fp32")]
    precision: String,
    #[arg(long, default_value_t = 4)]
    intra_threads: usize,
    /// Cap the number of windows processed (0 = all); handy for a quick smoke run.
    #[arg(long, default_value_t = 0)]
    limit: usize,
    /// "multi-pass" (default, mirrors scripts/run_coreml.py: one extract()
    /// call for entities, then one more per schema.relation_passes group) or
    /// "one-pass" (single combined schema, everything in one call).
    #[arg(long, default_value = "multi-pass")]
    mode: String,
}

/// Maps every byte offset that starts a Unicode scalar value to its char
/// (codepoint) index, plus `text.len() -> total char count`. Only byte
/// offsets that are themselves char boundaries are ever looked up (all of
/// ours come from word-boundary splitting), so continuation-byte entries are
/// left at 0 and never read.
fn byte_to_char_map(text: &str) -> Vec<usize> {
    let mut map = vec![0usize; text.len() + 1];
    let mut char_count = 0usize;
    for (byte_idx, _) in text.char_indices() {
        map[byte_idx] = char_count;
        char_count += 1;
    }
    map[text.len()] = char_count;
    map
}

#[derive(Debug, Deserialize)]
struct SchemaFile {
    entities: Vec<SchemaEntity>,
    /// v2: relations are grouped into passes (CoreML relation head is R4,
    /// bucket Q<=8), e.g. ["affiliation", "social_place"]. Every engine runs
    /// one extract() call per pass, on top of the entities call.
    relation_passes: Vec<String>,
    relations: Vec<SchemaRelation>,
    threshold: f32,
}

#[derive(Debug, Deserialize)]
struct SchemaEntity {
    ontonotes: String,
    label: String,
}

#[derive(Debug, Deserialize)]
struct SchemaRelation {
    /// Canonical output id (what we write to the shared format).
    name: String,
    /// Text sent to the model as the relation's prompt -- may differ from
    /// `name` (schema v2: name is e.g. "employee_of", prompt is "works for").
    prompt: String,
    /// Which relation_passes group this relation belongs to.
    pass: String,
}

#[derive(Debug, Deserialize)]
struct WindowRecord {
    window_id: serde_json::Value,
    text: String,
}

#[derive(Debug, Serialize)]
struct OutEntity {
    label: String,
    text: String,
    start: usize,
    end: usize,
    score: f32,
}

#[derive(Debug, Serialize)]
struct OutRelEndpoint {
    text: String,
    start: usize,
    end: usize,
}

#[derive(Debug, Serialize)]
struct OutRelation {
    name: String,
    head: OutRelEndpoint,
    tail: OutRelEndpoint,
    score: f32,
}

#[derive(Debug, Serialize)]
struct OutRecord {
    window_id: serde_json::Value,
    entities: Vec<OutEntity>,
    relations: Vec<OutRelation>,
    latency_ms: f64,
}

fn main() -> Result<()> {
    let args = Args::parse();

    // GLINER2_DEVICE / GLINER2_PRECISION are read by gliner25-rs at engine
    // build / load time, so they must be set before BoundaryConfig/BoundaryEngine
    // are constructed.
    std::env::set_var("GLINER2_DEVICE", &args.device);
    std::env::set_var("GLINER2_PRECISION", &args.precision);

    gliner25_rs::init("gliner-rust-eval");

    let schema_raw = std::fs::read_to_string(&args.schema)
        .with_context(|| format!("reading schema {}", args.schema.display()))?;
    let schema: SchemaFile = serde_json::from_str(&schema_raw)?;

    // label (gliner prompt string) -> OntoNotes tag, for mapping Mentions back
    let label_to_tag: std::collections::HashMap<String, String> = schema
        .entities
        .iter()
        .map(|e| (e.label.clone(), e.ontonotes.clone()))
        .collect();

    let entity_labels: Vec<String> = schema.entities.iter().map(|e| e.label.clone()).collect();
    let entity_tasks: Vec<SchemaTask> = vec![SchemaTask::Entities(entity_labels)];

    // prompt text sent to the model -> canonical output name (schema v2:
    // `name` e.g. "employee_of" vs `prompt` e.g. "works for").
    let prompt_to_name: std::collections::HashMap<String, String> = schema
        .relations
        .iter()
        .map(|r| (r.prompt.clone(), r.name.clone()))
        .collect();

    // One SchemaTask::Relations per relation, grouped by schema.relation_passes
    // (CoreML's relation head is R4 / bucket Q<=8, so relations run in groups
    // of <= 4). The prompt text -- not the canonical name -- is what becomes
    // the model's "[P]" marker per gliner25-rs's processor.rs (SchemaTask::Relations's
    // first field is literally sent as the prompt string).
    let relation_pass_tasks: Vec<Vec<SchemaTask>> = schema
        .relation_passes
        .iter()
        .map(|pass_name| {
            schema
                .relations
                .iter()
                .filter(|r| &r.pass == pass_name)
                .map(|r| {
                    SchemaTask::Relations(r.prompt.clone(), vec!["head".to_string(), "tail".to_string()])
                })
                .collect()
        })
        .collect();
    let all_relation_tasks: Vec<SchemaTask> = relation_pass_tasks.iter().flatten().cloned().collect();

    let multi_pass = match args.mode.as_str() {
        "multi-pass" => true,
        "one-pass" => false,
        other => anyhow::bail!("--mode must be 'one-pass' or 'multi-pass', got {other:?}"),
    };
    // one-pass: a single combined schema (8 entity labels + all relation
    // prompts across every pass) in one extract() call. multi-pass (default):
    // entities-only, then one call per relation_passes group, mirroring
    // scripts/run_coreml.py exactly -- see the module doc comment for why
    // this isn't just a latency knob.
    let mut combined_tasks: Vec<SchemaTask> = entity_tasks.clone();
    combined_tasks.extend(all_relation_tasks.clone());

    let params = BoundaryParams {
        threshold: schema.threshold,
        ..BoundaryParams::default()
    };

    eprintln!(
        "[gliner-rust-eval] loading model from {} (device={}, precision={})",
        args.models_dir.display(),
        args.device,
        args.precision
    );
    let load_start = Instant::now();
    let mut engine = BoundaryEngine::new(BoundaryConfig::new(&args.models_dir))?;
    let load_ms = load_start.elapsed().as_secs_f64() * 1000.0;
    eprintln!(
        "[gliner-rust-eval] model loaded in {load_ms:.1} ms; manifest: {:?} buckets, pool {}, overlap_policy={}",
        engine.manifest().length_buckets,
        engine.manifest().pool_size,
        engine.manifest().overlap_policy,
    );
    eprintln!(
        "[gliner-rust-eval] execution mode in force: {:?}",
        engine.execution()
    );

    let infile = File::open(&args.windows)
        .with_context(|| format!("opening windows file {}", args.windows.display()))?;
    let reader = BufReader::new(infile);
    let outfile = File::create(&args.out)
        .with_context(|| format!("creating output file {}", args.out.display()))?;
    let mut writer = BufWriter::new(outfile);

    let mut n = 0usize;
    let mut total_words = 0usize;
    let mut latencies_ms: Vec<f64> = Vec::new();

    for line in reader.lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let rec: WindowRecord = serde_json::from_str(&line)
            .with_context(|| format!("parsing window record: {line}"))?;

        // byte offset -> char offset for THIS window's text (see module doc
        // comment: Mention::char_start/char_end are actually byte offsets).
        let char_map = byte_to_char_map(&rec.text);

        let (ent_mentions, rel_pairs, elapsed_ms) = if multi_pass {
            let t0 = Instant::now();
            let ent_out = engine.extract_with(&rec.text, &entity_tasks, &params)?;
            let mut pairs = Vec::new();
            for pass_tasks in &relation_pass_tasks {
                if pass_tasks.is_empty() {
                    continue;
                }
                let rel_out = engine.extract_with(&rec.text, pass_tasks, &params)?;
                pairs.extend(pair_relations(&rel_out.mentions, pass_tasks));
            }
            let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;
            (ent_out.mentions, pairs, elapsed_ms)
        } else {
            let t0 = Instant::now();
            let out = engine.extract_with(&rec.text, &combined_tasks, &params)?;
            let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;
            let pairs = pair_relations(&out.mentions, &combined_tasks);
            (out.mentions, pairs, elapsed_ms)
        };
        latencies_ms.push(elapsed_ms);
        total_words += rec.text.split_whitespace().count();

        let entities: Vec<OutEntity> = ent_mentions
            .iter()
            .filter(|m| m.task == "entities")
            .map(|m| OutEntity {
                label: label_to_tag.get(&m.field).cloned().unwrap_or_else(|| m.field.clone()),
                text: m.text.clone(),
                start: char_map[m.char_start],
                end: char_map[m.char_end],
                score: m.score,
            })
            .collect();

        // pair_relations does a full cross-product of every head-role mention
        // with every tail-role mention sharing a relation name -- no joint
        // pair score. This is a genuine gliner25-rs gap, not shared with
        // gliner2: gliner2 2.0.0 decodes relation instances through a learned
        // scorer (gliner2/models/boundary/relations.py:93-230
        // TypedRelationPairGenerator, :320-409 SparseRelationScorer.forward,
        // score = self.mlp(feats) [+ biaffine + linear] at :376-408) fed by
        // *typed, capped* candidate pairs, not an unscored full cross-product.
        // gliner25-rs has no equivalent: the jugaadsrl/gliner2.5-multi-v1-onnx
        // export it loads ships no relation-scorer fragment at all (only
        // encoder/routed_gather/classifier/boundary_head_L*), unlike the
        // CoreML export's dedicated gliner2_multi_relation_*.mlpackage. We
        // report min(head.score, tail.score) as a stand-in pair score, purely
        // for the shared format -- it is not a model-provided quantity.
        let relations: Vec<OutRelation> = rel_pairs
            .into_iter()
            .map(|(h, t, prompt)| OutRelation {
                name: prompt_to_name.get(&prompt).cloned().unwrap_or(prompt),
                score: h.score.min(t.score),
                head: OutRelEndpoint {
                    text: h.text,
                    start: char_map[h.char_start],
                    end: char_map[h.char_end],
                },
                tail: OutRelEndpoint {
                    text: t.text,
                    start: char_map[t.char_start],
                    end: char_map[t.char_end],
                },
            })
            .collect();

        let out_rec = OutRecord {
            window_id: rec.window_id,
            entities,
            relations,
            latency_ms: elapsed_ms,
        };
        writeln!(writer, "{}", serde_json::to_string(&out_rec)?)?;

        n += 1;
        if args.limit > 0 && n >= args.limit {
            break;
        }
    }
    writer.flush()?;

    latencies_ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let p50 = percentile(&latencies_ms, 0.50);
    let p90 = percentile(&latencies_ms, 0.90);
    let total_ms: f64 = latencies_ms.iter().sum();
    let words_per_sec = if total_ms > 0.0 { total_words as f64 / (total_ms / 1000.0) } else { 0.0 };
    let peak_rss_mb = peak_rss_mb();

    eprintln!(
        "[gliner-rust-eval] windows={n} load_ms={load_ms:.1} p50_ms={p50:.2} p90_ms={p90:.2} \
         total_words={total_words} words_per_sec={words_per_sec:.1} peak_rss_mb={peak_rss_mb:.1}"
    );

    // machine-readable timing sidecar, next to the predictions file
    let timing_path = args.out.with_extension("timing.json");
    let timing = serde_json::json!({
        "engine": "rust-ort",
        "device": args.device,
        "precision": args.precision,
        "mode": args.mode,
        "execution_mode": format!("{:?}", engine.execution()),
        "windows": n,
        "load_ms": load_ms,
        "p50_latency_ms": p50,
        "p90_latency_ms": p90,
        "total_words": total_words,
        "words_per_sec": words_per_sec,
        "peak_rss_mb": peak_rss_mb,
    });
    std::fs::write(&timing_path, serde_json::to_string_pretty(&timing)?)?;
    eprintln!("[gliner-rust-eval] wrote timing sidecar to {}", timing_path.display());

    Ok(())
}

fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let idx = ((sorted.len() as f64 - 1.0) * p).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

#[cfg(target_os = "macos")]
fn peak_rss_mb() -> f64 {
    // macOS getrusage reports ru_maxrss in bytes (Linux reports KB -- this
    // binary only ever targets macOS per the project's environment notes).
    unsafe {
        let mut usage: libc::rusage = std::mem::zeroed();
        if libc::getrusage(libc::RUSAGE_SELF, &mut usage) == 0 {
            usage.ru_maxrss as f64 / (1024.0 * 1024.0)
        } else {
            0.0
        }
    }
}

#[cfg(not(target_os = "macos"))]
fn peak_rss_mb() -> f64 {
    unsafe {
        let mut usage: libc::rusage = std::mem::zeroed();
        if libc::getrusage(libc::RUSAGE_SELF, &mut usage) == 0 {
            usage.ru_maxrss as f64 / 1024.0
        } else {
            0.0
        }
    }
}
