//! pci-client: release pins, request, prover-side record preparation and the
//! fail-closed bundle verifier.
//!
//! ```text
//! pci-client rows    --model DIR --inventory INV --out rows.json
//! pci-client release --inventory INV --rows rows.json --shard-log2 S --setup-vk VK.json
//!                    --embed-vkey A.verkey.bin[,..] --layer-vkey B.verkey.bin[,..] --head-vkey C.verkey.bin[,..]
//!                    --out release.json
//! pci-client request --nonce HEX64 --prompt 1,2 --max-reply N --out request.json
//! pci-client prepare --model DIR --inventory INV --release R --request Q --reply 3,4 --a36 FILE --out DIR [--shards a-b]
//! pci-client proof   --release R --kind embed|layer|head --proof FILE [--digest HEX64]
//! pci-client assemble --expected expected.json --layers LAYER_BUNDLE.json --proofs DIR --out DIR
//!   (layer receipts from a pci-layer-bundle/v1 file; a child without NAME.proof.bin in DIR gets "proof": null)
//! pci-client check-prepared --release R --request Q --bundle DIR --digests ORDERED_HEX_LINES
//! pci-client verify  --release R --request Q --bundle DIR
//! pci-client verify-compact --release R --request Q --bundle DIR --recurser-vk VK.json --aggregate FILE
//!   (bundle.json receipts are used; per-child proofs are ignored; one aggregate proof)
//! ```
use pci_client::*;
use pci_receipts::{DIM, request_id};
use pci_stages::{embed, head, rows};
use serde_json::{Value, json};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

struct Opts(HashMap<String, String>);
impl Opts {
    fn parse(args: &[String]) -> Res<Self> {
        let mut m = HashMap::new();
        let mut it = args.iter();
        while let Some(k) = it.next() {
            let k = k.strip_prefix("--").ok_or(format!("unexpected argument {k}"))?;
            let v = it.next().ok_or(format!("--{k} needs a value"))?;
            if m.insert(k.to_string(), v.clone()).is_some() {
                return Err(format!("--{k} given twice"));
            }
        }
        Ok(Opts(m))
    }
    fn get(&self, k: &str) -> Res<&str> {
        self.0.get(k).map(String::as_str).ok_or(format!("missing --{k}"))
    }
    fn path(&self, k: &str) -> Res<PathBuf> {
        self.get(k).map(PathBuf::from)
    }
}

fn ids(s: &str) -> Res<Vec<u32>> {
    s.split(',')
        .filter(|x| !x.is_empty())
        .map(|x| x.trim().parse().map_err(|_| format!("bad id {x:?}")))
        .collect()
}

fn write_json(path: &Path, v: &Value) -> Res<()> {
    let tmp = path.with_extension("tmp");
    std::fs::write(&tmp, serde_json::to_vec_pretty(v).unwrap()).map_err(|e| e.to_string())?;
    std::fs::rename(&tmp, path).map_err(|e| e.to_string())
}

fn load_json(path: &Path) -> Res<Value> {
    serde_json::from_slice(&std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?)
        .map_err(|e| format!("{}: {e}", path.display()))
}

/// A ZisK stdin frame: u64 length, payload, zero padding to 8 bytes.
fn write_stdin(path: &Path, record: &[u8]) -> Res<()> {
    let mut f = (record.len() as u64).to_le_bytes().to_vec();
    f.extend_from_slice(record);
    f.resize(f.len().next_multiple_of(8), 0);
    std::fs::write(path, f).map_err(|e| e.to_string())
}

fn verkey_bin(path: &str) -> Res<[u64; 4]> {
    let b = std::fs::read(path).map_err(|e| format!("{path}: {e}"))?;
    if b.len() != 32 {
        return Err(format!("{path}: a verkey.bin is 32 bytes, got {}", b.len()));
    }
    let mut out = [0u64; 4];
    for (o, c) in out.iter_mut().zip(b.chunks_exact(8)) {
        *o = u64::from_le_bytes(c.try_into().unwrap());
    }
    Ok(out)
}

fn rows_cmd(o: &Opts) -> Res<()> {
    let inv = Inventory::load(&o.path("inventory")?)?;
    let ck = Checkpoint::open(&o.path("model")?)?;
    let t0 = std::time::Instant::now();
    let tied = ck.authenticated(&inv, TIED)?;
    let norm = ck.authenticated(&inv, NORM)?;
    let tree = rows::Tree::build(sha, &tied);
    let root = tree.root(sha);
    let v = json!({
        "format": "pci-rows/v1",
        "inventory_sha256": hex(&inv.file_sha256),
        "checkpoint": hex(&inv.checkpoint),
        "tied_sha256": hex(&inv.digest(TIED)?),
        "rows_root": hex(&root),
        "norm_sha256": hex(&inv.digest(NORM)?),
        "seconds": t0.elapsed().as_secs_f64(),
    });
    debug_assert_eq!(sha(&norm), inv.digest(NORM)?);
    write_json(&o.path("out")?, &v)?;
    println!("{}", serde_json::to_string_pretty(&v).unwrap());
    Ok(())
}

fn release_cmd(o: &Opts) -> Res<()> {
    let inv = Inventory::load(&o.path("inventory")?)?;
    let r = load_json(&o.path("rows")?)?;
    if r["format"] != "pci-rows/v1" || r["checkpoint"] != hex(&inv.checkpoint) {
        return Err("rows file is not for this inventory".into());
    }
    // An absent role list approves no program: every such child then rejects.
    let vks = |k: &str| -> Res<Vec<[u64; 4]>> {
        match o.0.get(k) {
            Some(v) => v.split(',').map(verkey_bin).collect(),
            None => Ok(vec![]),
        }
    };
    let setup: Value = load_json(&o.path("setup-vk")?)?;
    let setup_vk: [u64; 4] = serde_json::from_value(setup).map_err(|e| format!("setup-vk: {e}"))?;
    let pins = Pins {
        checkpoint: inv.checkpoint,
        stage_weights: inv.stage_weights()?,
        rows_root: unhex(r["rows_root"].as_str().ok_or("rows_root")?)?,
        norm_sha256: inv.digest(NORM)?,
        shard_log2: o.get("shard-log2")?.parse().map_err(|_| "shard-log2")?,
        context: 40960,
        stop_ids: vec![151643, 151645],
        setup_vk,
        embed_programs: vks("embed-vkey")?,
        layer_programs: vks("layer-vkey")?,
        head_programs: vks("head-vkey")?,
    };
    let v = pins.to_json();
    Pins::from_json(&v)?;
    write_json(&o.path("out")?, &v)?;
    println!("{}", serde_json::to_string_pretty(&v).unwrap());
    Ok(())
}

fn request_cmd(o: &Opts) -> Res<()> {
    let r = ClientRequest {
        nonce: unhex(o.get("nonce")?)?,
        prompt: ids(o.get("prompt")?)?,
        max_reply: o.get("max-reply")?.parse().map_err(|_| "max-reply")?,
    };
    write_json(&o.path("out")?, &r.to_json())
}

/// Prover side (untrusted): builds the embedding and head-shard records for a
/// finished transcript and the A36 the layer chain produced, and computes the
/// receipts natively (what the guests must reproduce).
fn prepare_cmd(o: &Opts) -> Res<()> {
    let inv = Inventory::load(&o.path("inventory")?)?;
    let pins = Pins::from_json(&load_json(&o.path("release")?)?)?;
    let req = ClientRequest::from_json(&load_json(&o.path("request")?)?)?;
    let reply = ids(o.get("reply")?)?;
    let (_, r, t, _) = check_request(&pins, &req, &reply).map_err(|e| format!("{e:?}"))?;
    let ck = Checkpoint::open(&o.path("model")?)?;
    let tied = ck.authenticated(&inv, TIED)?;
    let norm = ck.authenticated(&inv, NORM)?;
    let tree = rows::Tree::build(sha, &tied);
    if tree.root(sha) != pins.rows_root {
        return Err("row root differs from the release".into());
    }
    let release = pins.release();
    let (model, suite) = (release.model(), release.suite);
    let request = request_id(&release, &req.as_request());
    let out = o.path("out")?;
    std::fs::create_dir_all(&out).map_err(|e| e.to_string())?;
    let mut consumed = req.prompt.clone();
    consumed.extend_from_slice(&reply[..reply.len() - 1]);

    let rec = embed::encode(model, suite, request, &consumed, &tied, &tree);
    let e = embed::execute(&rec, sha)?;
    write_stdin(&out.join("embed.stdin"), &rec)?;
    let mut expected = vec![json!({"name": "embed", "digest": hex(&e.digest), "receipt": receipt_to_json(&e.receipt)})];
    std::fs::write(out.join("a0.f32"), &e.output).map_err(|e| e.to_string())?;

    let a36_bytes = std::fs::read(o.path("a36")?).map_err(|e| e.to_string())?;
    if a36_bytes.len() != t as usize * DIM as usize * 4 {
        return Err(format!("a36: expected {t} positions"));
    }
    let a36: Vec<f32> = a36_bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect();
    let n = head::shard_count(pins.shard_log2);
    let (lo, hi) = match o.0.get("shards") {
        Some(s) => {
            let (a, b) = s.split_once('-').ok_or("--shards a-b")?;
            (a.parse().map_err(|_| "shards")?, b.parse::<u32>().map_err(|_| "shards")? + 1)
        }
        None => (0, n),
    };
    let mut shard_receipts = vec![];
    for k in lo..hi.min(n) {
        let rec = head::encode(model, suite, request, pins.shard_log2, k, r, &a36, &norm, &tied, &tree);
        let h = head::execute(&rec, sha)?;
        let name = format!("head-{k:05}");
        write_stdin(&out.join(format!("{name}.stdin")), &rec)?;
        expected.push(json!({"name": name, "digest": hex(&h.digest), "receipt": shard_to_json(&h.receipt)}));
        shard_receipts.push(h.receipt);
    }
    let mut info = json!({
        "model": hex(&model), "suite": hex(&suite), "request": hex(&request),
        "positions": t, "predictions": r, "reply": reply,
        "a0_root": hex(&e.receipt.output),
        "a36_root": hex(&pci_receipts::activation_root(model, request, 36, t, &a36).map_err(|e| format!("{e:?}"))?),
        "children": expected,
    });
    if lo == 0 && hi >= n {
        let predicted = head::combine(&shard_receipts)?;
        info["predicted"] = json!(predicted);
        info["predicted_matches_reply"] = json!(predicted == reply);
    }
    write_json(&out.join("expected.json"), &info)?;
    eprintln!("prepared {} records in {}", info["children"].as_array().unwrap().len(), out.display());
    Ok(())
}

fn proof_cmd(o: &Opts) -> Res<()> {
    let pins = Pins::from_json(&load_json(&o.path("release")?)?)?;
    let kind = match o.get("kind")? {
        "embed" => Kind::Embed,
        "layer" => Kind::Layer,
        "head" => Kind::Head,
        k => return Err(format!("kind {k}")),
    };
    let proof = std::fs::read(o.path("proof")?).map_err(|e| e.to_string())?;
    let digest = Zisk { pins: &pins }.verify(kind, &proof)?;
    if let Ok(want) = o.get("digest") {
        if unhex(want)? != digest {
            return Err(format!("proof verified but its digest {} is not {want}", hex(&digest)));
        }
    }
    println!(
        "CHILD PROOF VERIFIED (single child; this is not a bundle acceptance): kind {:?} digest {}",
        kind,
        hex(&digest)
    );
    Ok(())
}

// Checks the expected emulator publics against the client's own request before paid work.
fn check_prepared_cmd(o: &Opts) -> Res<()> {
    let pins = Pins::from_json(&load_json(&o.path("release")?)?)?;
    let req = ClientRequest::from_json(&load_json(&o.path("request")?)?)?;
    let bundle = Bundle::load(&o.path("bundle")?)?;
    let digests = std::fs::read_to_string(o.path("digests")?).map_err(|e| e.to_string())?;
    let expected: Vec<_> = digests.lines().map(unhex).collect::<Res<_>>()?;
    if expected.len() != bundle.children.len() {
        return Err("prepared digest count differs from the receipt count".into());
    }
    let receipts: Vec<_> = bundle.children.iter().map(|child| &child.receipt).collect();
    let statement = join(&pins, &req, &bundle.reply, &receipts, &mut |i, _, digest| {
        if expected[i] != *digest {
            return Err(Reject::Receipt(i, "prepared digest mismatch".into()));
        }
        Ok(())
    })
    .map_err(|e| format!("receipt preflight: {e:?}"))?;
    println!("Prepared {} receipts for request {} (proofs pending)", receipts.len(), hex(&statement.request_id));
    Ok(())
}

fn verify_cmd(o: &Opts) -> Res<()> {
    let pins = Pins::from_json(&load_json(&o.path("release")?)?)?;
    let req = ClientRequest::from_json(&load_json(&o.path("request")?)?)?;
    let bundle = Bundle::load(&o.path("bundle")?)?;
    let t0 = std::time::Instant::now();
    match verify_bundle(&Zisk { pins: &pins }, &pins, &req, &bundle) {
        Ok(s) => {
            println!(
                "{}",
                serde_json::to_string_pretty(&json!({
                    "accepted": true, "model": hex(&s.model), "suite": hex(&s.suite),
                    "request": hex(&s.request_id), "reply": bundle.reply, "reply_root": hex(&s.reply_root),
                    "end": if s.stopped { "stop" } else { "length" },
                    "seconds": t0.elapsed().as_secs_f64(),
                }))
                .unwrap()
            );
            Ok(())
        }
        Err(e) => Err(format!("REJECTED: {e:?}")),
    }
}

fn verify_compact_cmd(o: &Opts) -> Res<()> {
    let pins = Pins::from_json(&load_json(&o.path("release")?)?)?;
    let req = ClientRequest::from_json(&load_json(&o.path("request")?)?)?;
    let bundle = Bundle::load(&o.path("bundle")?)?;
    let vk: [u64; 4] = serde_json::from_value(load_json(&o.path("recurser-vk")?)?).map_err(|e| e.to_string())?;
    let agg = std::fs::read(o.path("aggregate")?).map_err(|e| e.to_string())?;
    let receipts: Vec<ChildReceipt> = bundle.children.into_iter().map(|c| c.receipt).collect();
    let s = verify_compact(&pins, &vk, &req, &bundle.reply, &receipts, &agg).map_err(|e| format!("REJECTED: {e:?}"))?;
    println!("{}", json!({"accepted": true, "format": "compact", "request": hex(&s.request_id), "reply": s.predicted}));
    Ok(())
}

/// Prover side: bundle.json from the client-stage receipts (`prepare`), the 36
/// layer receipts of a `pci-layer-bundle/v1`, and whichever proofs exist.
fn assemble_cmd(o: &Opts) -> Res<()> {
    let e = load_json(&o.path("expected")?)?;
    let l = load_json(&o.path("layers")?)?;
    if l["format"] != "pci-layer-bundle/v1" {
        return Err("layers: not a pci-layer-bundle/v1".into());
    }
    for (field, layer_field) in [("model", "model_id"), ("suite", "suite_id"), ("request", "request_id")] {
        let expected = e[field].as_str().ok_or_else(|| format!("missing {field}"))?;
        if l["info"][layer_field].as_str() != Some(expected) {
            return Err(format!("layer bundle and head records disagree on {field}"));
        }
    }
    if l["reply"] != e["reply"] {
        return Err("layer bundle and head records disagree on reply".into());
    }
    let proofs = o.path("proofs")?;
    let out = o.path("out")?;
    std::fs::create_dir_all(&out).map_err(|e| e.to_string())?;
    let mut children = vec![];
    let mut missing = vec![];
    let mut add = |name: String, receipt: Value| -> Res<()> {
        let f = format!("{name}.proof.bin");
        let src = proofs.join(&f);
        if src.exists() {
            if out.join(&f) != src {
                std::fs::copy(&src, out.join(&f)).map_err(|e| e.to_string())?;
            }
            children.push((receipt, Value::String(f)));
        } else {
            missing.push(name);
            children.push((receipt, Value::Null));
        }
        Ok(())
    };
    let ec = e["children"].as_array().ok_or("expected children")?;
    add("embed".into(), ec[0]["receipt"].clone())?;
    let layers: Vec<&Value> = l["children"].as_array().ok_or("layer children")?.iter().filter(|c| c["kind"] == 1).collect();
    if layers.len() != 36 {
        return Err(format!("{} layer receipts, need 36", layers.len()));
    }
    for (i, c) in layers.iter().enumerate() {
        let mut r = serde_json::Map::new();
        r.insert("kind".into(), json!("layer"));
        for k in ["index", "positions", "first_prediction", "predictions", "input", "output", "weights"] {
            r.insert(k.into(), c[k].clone());
        }
        add(format!("layer{i:02}"), Value::Object(r))?;
    }
    for c in &ec[1..] {
        add(c["name"].as_str().ok_or("name")?.to_string(), c["receipt"].clone())?;
    }
    let v = json!({
        "format": "pci-proof-bundle/v1",
        "reply": e["reply"],
        "children": children.iter().map(|(r, p)| json!({"receipt": r, "proof": p})).collect::<Vec<_>>(),
    });
    write_json(&out.join("bundle.json"), &v)?;
    eprintln!("bundle: {} children, {} without proof", children.len(), missing.len());
    Ok(())
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let run = || -> Res<()> {
        let cmd = args.get(1).ok_or("usage: pci-client <command> --flags (see source header)")?;
        let o = Opts::parse(&args[2..])?;
        match cmd.as_str() {
            "rows" => rows_cmd(&o),
            "release" => release_cmd(&o),
            "request" => request_cmd(&o),
            "prepare" => prepare_cmd(&o),
            "proof" => proof_cmd(&o),
            "verify" => verify_cmd(&o),
            "check-prepared" => check_prepared_cmd(&o),
            "assemble" => assemble_cmd(&o),
            "verify-compact" => verify_compact_cmd(&o),
            c => Err(format!("unknown command {c}")),
        }
    };
    if let Err(e) = run() {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}
