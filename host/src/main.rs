//! qwen-host --model DIR --tokens 1,2,3 --elf PATH --ziskemu PATH --out DIR
//!           [--segments embed,0-35,head] [--flags N] [--emu-args "..."] [--keep 1]
//!           [--n-out N] [--dot exact] [--quant e4m3|e4m3-row] [--attn8 1]
//!
//! Chains the segments natively and runs each one in ziskemu, checking the
//! committed sha256 of its output against the native one.

use std::collections::HashMap;
use std::error::Error;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::process::Command;

use qwen_common::{
    ATTN8, DOTF8, EMBED, EXACT_FALLBACKS, GROUP, HEAD, HEADER_BYTES, Header, LAYER, ROWSCALE,
    argmax, dot_exact, dot_soft, qdot_soft, quantize, run,
};
use serde_json::Value;
use sha2::{Digest, Sha256};

type Result<T> = std::result::Result<T, Box<dyn Error>>;

struct Weights {
    dir: PathBuf,
    map: HashMap<String, (String, u64, u64)>,
}

impl Weights {
    fn open(dir: &Path) -> Result<Self> {
        let index: Value =
            serde_json::from_slice(&std::fs::read(dir.join("model.safetensors.index.json"))?)?;
        let mut shards = HashMap::new();
        for shard in index["weight_map"].as_object().ok_or("bad index")?.values() {
            let shard = shard.as_str().unwrap().to_string();
            if shards.contains_key(&shard) {
                continue;
            }
            let mut f = File::open(dir.join(&shard))?;
            let mut len = [0u8; 8];
            f.read_exact(&mut len)?;
            let mut header = vec![0u8; u64::from_le_bytes(len) as usize];
            f.read_exact(&mut header)?;
            shards.insert(
                shard,
                (
                    8 + header.len() as u64,
                    serde_json::from_slice::<Value>(&header)?,
                ),
            );
        }
        let mut map = HashMap::new();
        for (shard, (base, header)) in &shards {
            for (name, t) in header.as_object().unwrap() {
                if name == "__metadata__" {
                    continue;
                }
                assert_eq!(t["dtype"], "BF16", "{name}");
                let o = t["data_offsets"].as_array().unwrap();
                let (s, e) = (o[0].as_u64().unwrap(), o[1].as_u64().unwrap());
                map.insert(name.clone(), (shard.clone(), base + s, e - s));
            }
        }
        Ok(Weights {
            dir: dir.to_path_buf(),
            map,
        })
    }

    /// Appends tensor `name`, converted from bf16 to f32 (exact).
    fn append(&self, name: &str, out: &mut Vec<u8>) -> Result<()> {
        let (shard, at, len) = self
            .map
            .get(name)
            .ok_or_else(|| format!("no tensor {name}"))?;
        let mut f = File::open(self.dir.join(shard))?;
        f.seek(SeekFrom::Start(*at))?;
        let mut raw = vec![0u8; *len as usize];
        f.read_exact(&mut raw)?;
        out.reserve(raw.len() * 2);
        for b in raw.chunks_exact(2) {
            out.extend_from_slice(
                &(u32::from(u16::from_le_bytes([b[0], b[1]])) << 16).to_le_bytes(),
            );
        }
        Ok(())
    }

    /// Appends matrix `name` as E4M3 codes per group of `g`, then the groups' unit scales.
    fn append_q(&self, name: &str, g: usize, out: &mut Vec<u8>) -> Result<()> {
        let mut raw = Vec::new();
        self.append(name, &mut raw)?;
        let v: Vec<f32> = raw
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect();
        let mut codes = vec![0u8; v.len()];
        let scales: Vec<f32> = v
            .chunks(g)
            .zip(codes.chunks_mut(g))
            .map(|(g, c)| quantize(g, c))
            .collect();
        out.extend_from_slice(&codes);
        out.extend(scales.iter().flat_map(|s| s.to_le_bytes()));
        Ok(())
    }
}

fn floats(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn main() -> Result<()> {
    let mut opt: HashMap<String, String> = HashMap::new();
    let mut args = std::env::args().skip(1);
    while let Some(k) = args.next() {
        opt.insert(
            k.trim_start_matches("--").to_string(),
            args.next().ok_or("missing value")?,
        );
    }
    let get = |k: &str| opt.get(k).cloned().ok_or(format!("--{k} is required"));
    let model = PathBuf::from(get("model")?);
    let out_dir = PathBuf::from(get("out")?);
    let (elf, ziskemu) = (get("elf")?, get("ziskemu")?);
    let tokens: Vec<u32> = get("tokens")?
        .split(',')
        .map(str::parse)
        .collect::<std::result::Result<_, _>>()?;
    let (quant, row) = match opt.get("quant").map(String::as_str) {
        None => (false, false),
        Some("e4m3") => (true, false),
        Some("e4m3-row") => (true, true),
        Some(q) => return Err(format!("unknown --quant {q}").into()),
    };
    let flags: u32 = opt.get("flags").map_or(Ok(2), |f| f.parse())?
        | if quant { DOTF8 } else { 0 }
        | if row { ROWSCALE } else { 0 }
        | if opt.contains_key("attn8") { ATTN8 } else { 0 };
    let emu_args: Vec<String> = opt
        .get("emu-args")
        .map_or(vec![], |a| a.split_whitespace().map(String::from).collect());
    let dot: qwen_common::DotFn = match opt.get("dot").map(String::as_str) {
        None => dot_soft,
        Some("exact") => dot_exact,
        Some(d) => return Err(format!("unknown --dot {d}").into()),
    };
    std::fs::create_dir_all(&out_dir)?;

    let cfg: Value = serde_json::from_slice(&std::fs::read(model.join("config.json"))?)?;
    let n = |k: &str| cfg[k].as_u64().unwrap() as usize;
    let layers = n("num_hidden_layers");
    let base = Header {
        kind: 0,
        dim: n("hidden_size"),
        hidden: n("intermediate_size"),
        heads: n("num_attention_heads"),
        kv_heads: n("num_key_value_heads"),
        head_dim: n("head_dim"),
        vocab: n("vocab_size"),
        n_tok: tokens.len(),
        rope_theta: cfg["rope_theta"].as_f64().unwrap() as f32,
        eps: cfg["rms_norm_eps"].as_f64().unwrap() as f32,
        flags,
        n_out: opt.get("n-out").map_or(Ok(1), |v| v.parse())?,
    };
    assert!(
        cfg["tie_word_embeddings"].as_bool().unwrap(),
        "untied classifier"
    );
    let w = Weights::open(&model)?;

    let wanted = opt
        .get("segments")
        .cloned()
        .unwrap_or(format!("embed,0-{},head", layers - 1));
    let mut run_seg = vec![false; layers + 2];
    for part in wanted.split(',') {
        match part {
            "none" => {}
            "embed" => run_seg[0] = true,
            "head" => run_seg[layers + 1] = true,
            r => {
                let (a, b) = r.split_once('-').unwrap_or((r, r));
                for l in a.parse::<usize>()?..=b.parse::<usize>()? {
                    run_seg[l + 1] = true;
                }
            }
        }
    }

    let mut x: Vec<f32> = Vec::new();
    for seg in 0..layers + 2 {
        let (name, kind) = match seg {
            0 => ("embed".to_string(), EMBED),
            s if s == layers + 1 => ("head".to_string(), HEAD),
            s => (format!("layer{:02}", s - 1), LAYER),
        };
        let mut input = Header { kind, ..base }.to_bytes().to_vec();
        assert_eq!(input.len(), HEADER_BYTES);
        match kind {
            EMBED => {
                let mut t = tokens.clone();
                t.resize(t.len().next_multiple_of(2), 0);
                input.extend(t.iter().flat_map(|v| v.to_le_bytes()));
                w.append("model.embed_tokens.weight", &mut input)?;
            }
            LAYER => {
                input.extend(floats(&x));
                let p = format!("model.layers.{}.", seg - 1);
                for t in [
                    "input_layernorm",
                    "self_attn.q_proj",
                    "self_attn.k_proj",
                    "self_attn.v_proj",
                    "self_attn.o_proj",
                    "self_attn.q_norm",
                    "self_attn.k_norm",
                    "post_attention_layernorm",
                    "mlp.gate_proj",
                    "mlp.up_proj",
                    "mlp.down_proj",
                ] {
                    let name = format!("{p}{t}.weight");
                    if quant && t.ends_with("_proj") {
                        let cols = match t {
                            "self_attn.o_proj" => base.q_dim(),
                            "mlp.down_proj" => base.hidden,
                            _ => base.dim,
                        };
                        w.append_q(&name, if row { cols } else { GROUP }, &mut input)?;
                    } else {
                        w.append(&name, &mut input)?;
                    }
                }
            }
            _ => {
                input.extend(floats(&x));
                w.append("model.norm.weight", &mut input)?;
                if quant {
                    let g = if row { base.dim } else { GROUP };
                    w.append_q("model.embed_tokens.weight", g, &mut input)?;
                } else {
                    w.append("model.embed_tokens.weight", &mut input)?;
                }
            }
        }
        let native = run(&input, dot, qdot_soft);

        if run_seg[seg] {
            let stdin = out_dir.join(format!("{name}.stdin"));
            let mut rec = (input.len() as u64).to_le_bytes().to_vec();
            rec.extend_from_slice(&input);
            rec.resize(rec.len().next_multiple_of(8), 0);
            std::fs::write(&stdin, &rec)?;
            drop(rec);
            let committed = out_dir.join(format!("{name}.output"));
            let t = std::time::Instant::now();
            let emu = Command::new(&ziskemu)
                .args(["-e", &elf, "-i"])
                .arg(&stdin)
                .arg("-o")
                .arg(&committed)
                .args(&emu_args)
                .output()?;
            if !opt.contains_key("keep") {
                std::fs::remove_file(&stdin)?;
            }
            std::fs::write(
                out_dir.join(format!("{name}.txt")),
                [&emu.stdout[..], &emu.stderr[..]].concat(),
            )?;
            if !emu.status.success() {
                return Err(format!("{name}: ziskemu failed, see {name}.txt").into());
            }
            let c = std::fs::read(&committed)?;
            let ok = c.get(..32) == Some(&Sha256::digest(floats(&native))[..]);
            let text = String::from_utf8_lossy(&emu.stdout);
            let field = |k: &str| {
                text.lines()
                    .find_map(|l| l.trim().strip_prefix(k).map(|r| r.trim().to_string()))
            };
            println!(
                "{name} input {} MB, {:.1} s, steps {}, total cost {}, output matches native: {}",
                input.len() >> 20,
                t.elapsed().as_secs_f64(),
                field("STEPS").unwrap_or_default(),
                field("TOTAL").unwrap_or_default(),
                if ok { "yes" } else { "NO" }
            );
        }
        x = native;
    }

    let next: Vec<usize> = x.chunks(base.vocab).map(argmax).collect();
    println!("next tokens at the last {} positions: {next:?}", base.n_out);
    println!(
        "dot_exact fallbacks: {}",
        EXACT_FALLBACKS.load(std::sync::atomic::Ordering::Relaxed)
    );
    std::fs::write(out_dir.join("logits.f32"), floats(&x))?;
    Ok(())
}
