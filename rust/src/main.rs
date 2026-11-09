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

use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result};
use clap::Parser;
use serde::{Deserialize, Serialize};

use gliner25_rs::{
    BoundaryConfig, BoundaryEngine, BoundaryParams, Mention, SchemaTask, pair_relations,
};

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
}

#[derive(Debug, Deserialize)]
struct SchemaFile {
    entities: Vec<SchemaEntity>,
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
    name: String,
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
    let mut tasks: Vec<SchemaTask> = vec![SchemaTask::Entities(entity_labels)];
    for rel in &schema.relations {
        tasks.push(SchemaTask::Relations(
            rel.name.clone(),
            vec!["head".to_string(), "tail".to_string()],
        ));
    }

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

        let t0 = Instant::now();
        let out = engine.extract_with(&rec.text, &tasks, &params)?;
        let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;
        latencies_ms.push(elapsed_ms);
        total_words += rec.text.split_whitespace().count();

        let entities: Vec<OutEntity> = out
            .mentions
            .iter()
            .filter(|m| m.task == "entities")
            .map(|m| OutEntity {
                label: label_to_tag.get(&m.field).cloned().unwrap_or_else(|| m.field.clone()),
                text: m.text.clone(),
                start: m.char_start,
                end: m.char_end,
                score: m.score,
            })
            .collect();

        // pair_relations does a full cross-product of head x tail mentions per
        // relation group (gliner25-rs has no joint head/tail score -- neither
        // does gliner2's public `extract_relations`, which also returns bare
        // tuples). We report min(head.score, tail.score) as the pair score,
        // matching what scripts/run_torch.py does for the same reason.
        let rel_pairs: Vec<(Mention, Mention, String)> = pair_relations(&out.mentions, &tasks);
        let relations: Vec<OutRelation> = rel_pairs
            .into_iter()
            .map(|(h, t, name)| OutRelation {
                name,
                score: h.score.min(t.score),
                head: OutRelEndpoint { text: h.text, start: h.char_start, end: h.char_end },
                tail: OutRelEndpoint { text: t.text, start: t.char_start, end: t.char_end },
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
