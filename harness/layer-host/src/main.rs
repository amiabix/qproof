//! Prover-side bundle builder and client-side bundle verifier for the layer guest.
//!
//! pci-layer-host run    --model DIR --inventory INV.json --elf LAYER.elf --ziskemu EMU --out DIR
//!                       --prompt 1,2 [--reply 3,4] --max-reply N --nonce HEX64 [--emulate 0-35|none]
//!                       [--emu-args "..."] [--shard-log2 9] [--keep 1] [--dump 0-0] [--jobs N]
//! pci-layer-host verify --inventory INV.json --elf-sha256 HEX64 --bundle DIR
//!                       --prompt 1,2 --max-reply N --nonce HEX64 [--allow-native-embed-head 1]
//! pci-layer-host forge  --model DIR --inventory INV.json --elf LAYER.elf --ziskemu EMU --bundle DIR
//!                       --layer I --prompt ... --max-reply N --nonce HEX64 --out DIR
//!
//! The verifier takes ModelID, SuiteID and stage weight roots only from the
//! client's inventory and ELF pin, and the request only from its own arguments.
//! Layer evidence is the ZisK emulator's public output, not a proof.
use pci_layer::{TENSOR_LEN, encode, execute, widen};
use pci_receipts::*;
use qwen_common::{HEAD, HEADER_BYTES, Header, run_with};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::process::Command;

type Res<T> = std::result::Result<T, Box<dyn std::error::Error>>;

fn sha(b: &[u8]) -> Hash {
    Sha256::digest(b).into()
}
fn hex(h: &[u8]) -> String {
    h.iter().map(|b| format!("{b:02x}")).collect()
}
fn unhex(s: &str) -> Res<Hash> {
    if s.len() != 64 || !s.bytes().all(|b| b.is_ascii_hexdigit()) {
        return Err("expected 64 hex digits".into());
    }
    let mut h = [0u8; 32];
    for (i, b) in h.iter_mut().enumerate() {
        *b = u8::from_str_radix(&s[2 * i..2 * i + 2], 16)?;
    }
    Ok(h)
}
fn ids(s: &str) -> Res<Vec<u32>> {
    s.split(',')
        .filter(|t| !t.is_empty())
        .map(|t| Ok(t.trim().parse()?))
        .collect()
}

// ---------------------------------------------------------------- pins

/// Client pins: the enrollment inventory of the publisher checkpoint.
struct Pins {
    checkpoint: Hash,
    tensors: HashMap<String, Hash>,
}

impl Pins {
    fn load(path: &Path) -> Res<Self> {
        let v: Value = serde_json::from_slice(&std::fs::read(path)?)?;
        let checkpoint = unhex(v["checkpoint_id"].as_str().ok_or("checkpoint_id")?)?;
        // Recompute enroll.py's ID: compact, key-sorted JSON of everything else.
        let mut body = v.as_object().ok_or("inventory")?.clone();
        body.remove("checkpoint_id");
        let mut pre = b"PCI/checkpoint-inventory/v1\0".to_vec();
        pre.extend(serde_json::to_vec(&body)?);
        if sha(&pre) != checkpoint {
            return Err("checkpoint_id does not match the inventory contents".into());
        }
        let mut tensors = HashMap::new();
        for (name, t) in v["tensors"].as_object().ok_or("tensors")? {
            if t["dtype"] != "BF16" {
                return Err(format!("{name}: not BF16").into());
            }
            tensors.insert(name.clone(), unhex(t["sha256"].as_str().ok_or("sha256")?)?);
        }
        // The inventory's stage roster must be exactly the roster this verifier enforces.
        let stages = v["stages"].as_array().ok_or("stages")?;
        if stages.len() != STAGES {
            return Err("inventory stage count".into());
        }
        for (i, s) in stages.iter().enumerate() {
            let listed: Vec<&str> = s["tensors"]
                .as_array()
                .ok_or("stage tensors")?
                .iter()
                .map(|t| t.as_str().unwrap_or(""))
                .collect();
            if listed != stage_tensors(role(i)) {
                return Err(format!("inventory stage {i} roster differs").into());
            }
        }
        Ok(Pins {
            checkpoint,
            tensors,
        })
    }

    fn stage_root(&self, r: Role) -> Res<Hash> {
        let d = stage_tensors(r)
            .iter()
            .map(|n| self.tensors.get(n).copied().ok_or(format!("unpinned {n}")))
            .collect::<std::result::Result<Vec<_>, _>>()?;
        Ok(weights_root(r, &d))
    }

    fn release(&self, o: &Opts) -> Res<Release> {
        Ok(Release {
            checkpoint: self.checkpoint,
            weights: (0..STAGES)
                .map(|i| self.stage_root(role(i)))
                .collect::<Res<_>>()?,
            suite: pci_receipts::suite_id(o.0.get("shard-log2").map_or(Ok(9), |s| s.parse())?),
            context: 40960,
            stop_ids: vec![151643, 151645],
        })
    }
}

fn role(i: usize) -> Role {
    match i {
        0 => Role::Embed,
        i if i <= LAYERS => Role::Layer(i as u32 - 1),
        _ => Role::Head,
    }
}

// ---------------------------------------------------------------- checkpoint

struct Checkpoint {
    dir: PathBuf,
    map: HashMap<String, (String, u64, u64)>,
}

impl Checkpoint {
    fn open(dir: &Path) -> Res<Self> {
        let index: Value =
            serde_json::from_slice(&std::fs::read(dir.join("model.safetensors.index.json"))?)?;
        let mut map = HashMap::new();
        let mut shards: Vec<String> = index["weight_map"]
            .as_object()
            .ok_or("index")?
            .values()
            .filter_map(|v| v.as_str().map(String::from))
            .collect();
        shards.sort();
        shards.dedup();
        for shard in shards {
            let mut f = File::open(dir.join(&shard))?;
            let mut len = [0u8; 8];
            f.read_exact(&mut len)?;
            let mut header = vec![0u8; u64::from_le_bytes(len) as usize];
            f.read_exact(&mut header)?;
            let header: Value = serde_json::from_slice(&header)?;
            let base = 8 + u64::from_le_bytes(len);
            for (name, t) in header.as_object().ok_or("shard header")? {
                if name == "__metadata__" {
                    continue;
                }
                let o = t["data_offsets"].as_array().ok_or("offsets")?;
                let (s, e) = (o[0].as_u64().unwrap(), o[1].as_u64().unwrap());
                map.insert(name.clone(), (shard.clone(), base + s, e - s));
            }
        }
        Ok(Checkpoint {
            dir: dir.to_path_buf(),
            map,
        })
    }

    /// Raw checkpoint bytes. Untrusted: the guest and the verifier authenticate them.
    fn raw(&self, name: &str) -> Res<Vec<u8>> {
        let (shard, at, len) = self.map.get(name).ok_or(format!("no tensor {name}"))?;
        let mut f = File::open(self.dir.join(shard))?;
        f.seek(SeekFrom::Start(*at))?;
        let mut v = vec![0u8; *len as usize];
        f.read_exact(&mut v)?;
        Ok(v)
    }
}

// ---------------------------------------------------------------- bundle I/O

fn receipt_json(r: &Receipt, evidence: Value) -> Value {
    let (kind, index) = match r.role {
        Role::Embed => (0, 0),
        Role::Layer(i) => (1, i),
        Role::Head => (2, 0),
    };
    json!({"kind": kind, "index": index, "positions": r.positions,
        "first_prediction": r.first_prediction, "predictions": r.predictions,
        "input": hex(&r.input), "output": hex(&r.output), "weights": hex(&r.weights),
        "evidence": evidence})
}

fn receipt_from(v: &Value) -> Res<Receipt> {
    let n = |k: &str| -> Res<u32> { Ok(u32::try_from(v[k].as_u64().ok_or(k.to_string())?)?) };
    let h = |k: &str| unhex(v[k].as_str().ok_or(k.to_string())?);
    let role = match (n("kind")?, n("index")?) {
        (0, 0) => Role::Embed,
        (1, i) => Role::Layer(i),
        (2, 0) => Role::Head,
        _ => return Err("bad role".into()),
    };
    Ok(Receipt {
        role,
        positions: n("positions")?,
        first_prediction: n("first_prediction")?,
        predictions: n("predictions")?,
        input: h("input")?,
        output: h("output")?,
        weights: h("weights")?,
    })
}

/// Evidence a child carries. Neither variant is a succinct proof.
enum Evidence {
    /// ZisK emulator public output file, and the layer ELF the prover says it ran.
    Emulated(PathBuf, Hash),
    /// Digest asserted by the host for a stage that has no guest yet.
    Native(Hash),
}

struct EmulatorTranscript {
    allow_native: bool,
    /// Approved layer program. In this transcript mode the program identity is
    /// the prover's statement; a proof backend must instead compare the
    /// proof's program VK / ROM root with the pinned value.
    program: Hash,
}

impl ProofBackend for EmulatorTranscript {
    type Proof = Evidence;
    fn verify_child(&self, _suite: &Hash, expected: Role, e: &Evidence) -> Result<Hash, Error> {
        match (expected, e) {
            (Role::Layer(_), Evidence::Emulated(path, elf)) => {
                if *elf != self.program {
                    return Err(Error::Proof);
                }
                let out = std::fs::read(path).map_err(|_| Error::Proof)?;
                // ABI: a 32-byte receipt digest followed by zero padding.
                if out.len() < 32 || out[32..].iter().any(|b| *b != 0) {
                    return Err(Error::Proof);
                }
                Ok(out[..32].try_into().unwrap())
            }
            (Role::Embed | Role::Head, Evidence::Native(d)) if self.allow_native => Ok(*d),
            _ => Err(Error::Proof),
        }
    }
}

fn load_bundle(dir: &Path) -> Res<(Vec<u32>, Vec<(Receipt, Evidence)>)> {
    let v: Value = serde_json::from_slice(&std::fs::read(dir.join("bundle.json"))?)?;
    let reply = v["reply"]
        .as_array()
        .ok_or("reply")?
        .iter()
        .map(|x| Ok(u32::try_from(x.as_u64().ok_or("id")?)?))
        .collect::<Res<Vec<u32>>>()?;
    let mut children = Vec::new();
    for c in v["children"].as_array().ok_or("children")? {
        let e = &c["evidence"];
        let evidence = match e["type"].as_str() {
            Some("emulator-output") => {
                let f = e["file"].as_str().ok_or("file")?;
                // A plain regular file inside the bundle: no separators, dot names or symlinks.
                let plain = !f.is_empty()
                    && f.bytes().all(|b| b.is_ascii_alphanumeric() || b"._-".contains(&b))
                    && !f.starts_with('.');
                if !plain || !std::fs::symlink_metadata(dir.join(f))?.file_type().is_file() {
                    return Err("evidence must be a regular file in the bundle".into());
                }
                Evidence::Emulated(
                    dir.join(f),
                    unhex(e["elf_sha256"].as_str().ok_or("elf_sha256")?)?,
                )
            }
            Some("native-claim") => Evidence::Native(unhex(e["digest"].as_str().ok_or("digest")?)?),
            _ => return Err("unknown evidence".into()),
        };
        children.push((receipt_from(c)?, evidence));
    }
    Ok((reply, children))
}

// ---------------------------------------------------------------- native stages

fn embed(tied: &[u8], consumed: &[u32]) -> Res<Vec<f32>> {
    let mut x = Vec::with_capacity(consumed.len() * DIM as usize);
    for &t in consumed {
        if t >= VOCAB {
            return Err("token out of range".into());
        }
        let row = 2 * DIM as usize * t as usize;
        x.extend(widen(&tied[row..row + 2 * DIM as usize]));
    }
    Ok(x)
}

/// Greedy IDs at the last `n_out` positions under the pinned BF16 serving profile, lowest ID on ties.
fn head(x: &[f32], norm: &[u8], tied: &[u8], n_out: usize) -> (Vec<u32>, Vec<f32>) {
    let t = x.len() / DIM as usize;
    let h = Header {
        kind: HEAD,
        dim: 2560,
        hidden: 9728,
        heads: 32,
        kv_heads: 8,
        head_dim: 128,
        vocab: VOCAB as usize,
        n_tok: t,
        rope_theta: 1e6,
        eps: 1e-6,
        flags: qwen_common::BF16_SERVING,
        n_out,
    };
    let mut seg: Vec<u32> = h
        .to_bytes()
        .chunks_exact(4)
        .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
        .collect();
    assert_eq!(seg.len() * 4, HEADER_BYTES);
    seg.extend(x.iter().map(|v| v.to_bits()));
    seg.extend(widen(norm).map(f32::to_bits));
    let bytes = unsafe { std::slice::from_raw_parts(seg.as_ptr().cast::<u8>(), 4 * seg.len()) };
    let xn = std::cell::RefCell::new(Vec::new());
    let mut obs = |_: usize, v: &[f32]| *xn.borrow_mut() = v.to_vec();
    let mut claim = |_: usize, out: &mut [f32]| pci_layer::project_bf16(&xn.borrow(), tied, out);
    let logits = run_with(
        bytes,
        qwen_common::dot_soft,
        qwen_common::qdot_soft,
        &mut obs,
        Some(&mut claim),
    );
    let ids = logits
        .chunks(VOCAB as usize)
        .map(|l| qwen_common::argmax(l) as u32)
        .collect();
    (ids, logits)
}

// ---------------------------------------------------------------- emulator

struct Emu {
    elf: PathBuf,
    ziskemu: PathBuf,
    args: Vec<String>,
}

struct Job {
    name: String,
    child: std::process::Child,
    stdin: PathBuf,
    output: PathBuf,
    log: PathBuf,
    start: std::time::Instant,
    keep: bool,
}

impl Emu {
    /// Writes ZisK stdin and starts the pinned ELF on it.
    fn start(&self, out: &Path, name: &str, record: &[u8], keep: bool) -> Res<Job> {
        let stdin = out.join(format!("{name}.stdin"));
        let mut rec = (record.len() as u64).to_le_bytes().to_vec();
        rec.extend_from_slice(record);
        rec.resize(rec.len().next_multiple_of(8), 0);
        std::fs::write(&stdin, &rec)?;
        let input_limit = rec.len() + 8; // ZisK reserves one header word.
        drop(rec);
        let output = out.join(format!("{name}.output"));
        let _ = std::fs::remove_file(&output);
        let log = out.join(format!("{name}.txt"));
        let f = File::create(&log)?;
        let mut cmd = Command::new(&self.ziskemu);
        if !self.args.iter().any(|a| a == "--max-input-mem" || a.starts_with("--max-input-mem=")) {
            cmd.arg("--max-input-mem").arg(input_limit.to_string());
        }
        let child = cmd
            .arg("-e")
            .arg(&self.elf)
            .arg("-i")
            .arg(&stdin)
            .arg("-o")
            .arg(&output)
            .args(&self.args)
            .stdout(f.try_clone()?)
            .stderr(f)
            .spawn()?;
        Ok(Job {
            name: name.to_string(),
            child,
            stdin,
            output,
            log,
            start: std::time::Instant::now(),
            keep,
        })
    }

    /// Waits; returns (success, steps, output file).
    fn finish(mut job: Job) -> Res<(bool, String, PathBuf)> {
        let status = job.child.wait()?;
        if !job.keep {
            std::fs::remove_file(&job.stdin)?;
        }
        let text = std::fs::read_to_string(&job.log).unwrap_or_default();
        let steps = text
            .lines()
            .find_map(|line| {
                let line = line.trim();
                line.strip_prefix("STEPS").map(str::trim).or_else(|| {
                    line.strip_prefix("process_rom() steps=")?.split_whitespace().next()
                }).map(str::to_owned)
            })
            .unwrap_or_default();
        eprintln!(
            "{}: emulator exit {:?}, {:.1} s, steps {steps}",
            job.name,
            status.code(),
            job.start.elapsed().as_secs_f64()
        );
        Ok((status.success() && job.output.exists(), steps, job.output))
    }
}

// ---------------------------------------------------------------- commands

struct Opts(HashMap<String, String>);
impl Opts {
    fn parse() -> Res<(String, Self)> {
        let mut a = std::env::args().skip(1);
        let cmd = a.next().ok_or("command")?;
        let mut m = HashMap::new();
        while let Some(k) = a.next() {
            m.insert(
                k.trim_start_matches("--").to_string(),
                a.next().ok_or("missing value")?,
            );
        }
        Ok((cmd, Opts(m)))
    }
    fn get(&self, k: &str) -> Res<&str> {
        self.0
            .get(k)
            .map(String::as_str)
            .ok_or(format!("missing --{k}").into())
    }
    /// Boolean option: absent or "0" is false, "1" is true, anything else is an error.
    fn flag(&self, k: &str) -> Res<bool> {
        match self.0.get(k).map(String::as_str) {
            None | Some("0") => Ok(false),
            Some("1") => Ok(true),
            Some(v) => Err(format!("--{k} takes 0 or 1, not {v}").into()),
        }
    }
    fn path(&self, k: &str) -> Res<PathBuf> {
        Ok(PathBuf::from(self.get(k)?))
    }
}

struct Req {
    prompt: Vec<u32>,
    max_reply: u32,
    nonce: Hash,
}
impl Req {
    fn from(o: &Opts) -> Res<Self> {
        Ok(Req {
            prompt: ids(o.get("prompt")?)?,
            max_reply: o.get("max-reply")?.parse()?,
            nonce: unhex(o.get("nonce")?)?,
        })
    }
    fn as_request(&self) -> Request<'_> {
        Request {
            nonce: self.nonce,
            prompt: &self.prompt,
            max_reply: self.max_reply,
        }
    }
}

fn layer_tensors(ck: &Checkpoint, i: u32) -> Res<Vec<Vec<u8>>> {
    let t: Vec<Vec<u8>> = stage_tensors(Role::Layer(i))
        .iter()
        .map(|n| ck.raw(n))
        .collect::<Res<_>>()?;
    for (b, n) in t.iter().zip(TENSOR_LEN) {
        if b.len() != 2 * n {
            return Err("tensor size".into());
        }
    }
    Ok(t)
}

fn range(s: &str) -> Res<Vec<u32>> {
    if s == "none" {
        return Ok(vec![]);
    }
    let (a, b) = s.split_once('-').unwrap_or((s, s));
    Ok((a.parse()?..=b.parse()?).collect())
}

fn run(o: &Opts) -> Res<()> {
    let out = o.path("out")?;
    std::fs::create_dir_all(&out)?;
    let pins = Pins::load(&o.path("inventory")?)?;
    let ck = Checkpoint::open(&o.path("model")?)?;
    let elf = o.path("elf")?;
    let elf_sha = sha(&std::fs::read(&elf)?);
    let release = pins.release(o)?;
    let req = Req::from(o)?;
    let q = req.as_request();
    let model = release.model();
    let rq = request_id(&release, &q);
    let supplied = o.0.get("reply").map(|s| ids(s)).transpose()?;
    let feedback: Vec<u32> = match &supplied {
        Some(r) if !r.is_empty() => r[..r.len() - 1].to_vec(),
        _ => vec![],
    };
    let r_count = supplied.as_ref().map_or(1, Vec::len);
    let consumed: Vec<u32> = req.prompt.iter().chain(&feedback).copied().collect();
    let t = consumed.len() as u32;
    let emulate = range(o.0.get("emulate").map_or("0-35", String::as_str))?;
    let emu = Emu {
        elf,
        ziskemu: o.path("ziskemu")?,
        args: o
            .0
            .get("emu-args")
            .map(|s| s.split_whitespace().map(String::from).collect())
            .unwrap_or_default(),
    };
    let keep = o.flag("keep")?;
    let dump = range(o.0.get("dump").map_or("none", String::as_str))?;
    let jobs: usize = o.0.get("jobs").map_or(Ok(1), |j| j.parse())?;
    let mut pending = std::collections::VecDeque::new();
    let check = |i: u32, job: Job, digest: Hash| -> Res<Value> {
        let (ok, steps, file) = Emu::finish(job)?;
        let c = if ok { std::fs::read(&file)? } else { vec![] };
        let matches = c.get(..32) == Some(&digest[..]) && c[32..].iter().all(|b| *b == 0);
        if !matches {
            return Err(format!("layer {i}: emulator output differs from native").into());
        }
        Ok(json!({"layer": i, "steps": steps, "emulator_ok": ok, "matches_native": matches}))
    };

    let tied = ck.raw("model.embed_tokens.weight")?;
    let norm = ck.raw("model.norm.weight")?;
    // Fail early if the local checkpoint is not the pinned one.
    for (r, d) in [
        (Role::Embed, vec![sha(&tied)]),
        (Role::Head, vec![sha(&norm), sha(&tied)]),
    ] {
        if weights_root(r, &d) != pins.stage_root(r)? {
            return Err("local tied/norm tensors differ from the pins".into());
        }
    }
    let mut x = embed(&tied, &consumed)?;
    let mut children = Vec::new();
    let mut summary = Vec::new();
    let a0 = activation_root(model, rq, 0, t, &x).map_err(|e| format!("{e:?}"))?;
    let er = Receipt {
        role: Role::Embed,
        positions: t,
        first_prediction: 0,
        predictions: 0,
        input: consumed_root(&req.prompt, &feedback),
        output: a0,
        weights: release.weights[0],
    };
    let d = er.digest(model, release.suite, rq);
    children.push(receipt_json(
        &er,
        json!({"type": "native-claim", "digest": hex(&d)}),
    ));

    for i in 0..LAYERS as u32 {
        let tensors = layer_tensors(&ck, i)?;
        let record = encode(model, release.suite, rq, i, r_count as u32, &x, &tensors);
        drop(tensors);
        let native = execute(&record, sha).map_err(|e| format!("layer {i}: {e}"))?;
        let digest = pci_layer::digest_of(&record, &native.receipt);
        let name = format!("layer{i:02}");
        if dump.contains(&i) {
            let mut rec = (record.len() as u64).to_le_bytes().to_vec();
            rec.extend_from_slice(&record);
            rec.resize(rec.len().next_multiple_of(8), 0);
            std::fs::write(out.join(format!("{name}.v1.stdin")), rec)?;
            std::fs::write(out.join(format!("{name}.digest")), digest)?;
            std::fs::write(
                out.join(format!("{name}.output.f32")),
                native
                    .output
                    .iter()
                    .flat_map(|v| v.to_le_bytes())
                    .collect::<Vec<u8>>(),
            )?;
            std::fs::write(
                out.join(format!("{name}.json")),
                serde_json::to_vec_pretty(
                    &json!({"model_id": hex(&model), "suite_id": hex(&release.suite),
                    "request_id": hex(&rq), "layer": i, "positions": t, "digest": hex(&digest),
                    "receipt": receipt_json(&native.receipt, Value::Null)}),
                )?,
            )?;
        }
        let evidence = if emulate.contains(&i) {
            if pending.len() >= jobs {
                let (l, job, d) = pending.pop_front().unwrap();
                summary.push(check(l, job, d)?);
            }
            pending.push_back((i, emu.start(&out, &name, &record, keep)?, digest));
            json!({"type": "emulator-output", "file": format!("{name}.output"), "elf_sha256": hex(&elf_sha)})
        } else {
            json!({"type": "native-claim", "digest": hex(&digest)})
        };
        children.push(receipt_json(&native.receipt, evidence));
        x = native.output;
        eprintln!("layer {i} done");
    }

    while let Some((l, job, d)) = pending.pop_front() {
        summary.push(check(l, job, d)?);
    }
    let (predicted, logits) = head(&x, &norm, &tied, r_count);
    std::fs::write(
        out.join("logits.f32"),
        logits
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<u8>>(),
    )?;
    let reply = match supplied {
        Some(r) if r != predicted => {
            return Err(format!("supplied reply {r:?} is not greedy: {predicted:?}").into());
        }
        Some(r) => r,
        None => predicted,
    };
    let hr = Receipt {
        role: Role::Head,
        positions: t,
        first_prediction: req.prompt.len() as u32 - 1,
        predictions: reply.len() as u32,
        input: activation_root(model, rq, LAYERS as u32, t, &x).map_err(|e| format!("{e:?}"))?,
        output: reply_root(&reply),
        weights: release.weights[STAGES - 1],
    };
    let d = hr.digest(model, release.suite, rq);
    children.push(receipt_json(
        &hr,
        json!({"type": "native-claim", "digest": hex(&d)}),
    ));
    let bundle = json!({"format": "pci-layer-bundle/v1", "reply": reply, "children": children,
        "info": {"model_id": hex(&model), "suite_id": hex(&release.suite), "request_id": hex(&rq),
                  "layer_elf_sha256": hex(&elf_sha), "emulated": summary}});
    std::fs::write(out.join("bundle.json"), serde_json::to_vec_pretty(&bundle)?)?;
    std::fs::write(
        out.join("a36.f32"),
        x.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>(),
    )?;
    println!("reply {reply:?}; bundle written to {}", out.display());
    Ok(())
}

fn verify(o: &Opts) -> Res<()> {
    let pins = Pins::load(&o.path("inventory")?)?;
    let release = pins.release(o)?;
    let program = unhex(o.get("elf-sha256")?)?;
    let req = Req::from(o)?;
    let dir = o.path("bundle")?;
    let (reply, children) = load_bundle(&dir)?;
    let backend = EmulatorTranscript {
        allow_native: o.flag("allow-native-embed-head")?,
        program,
    };
    let native: Vec<String> = children
        .iter()
        .filter(|(_, e)| matches!(e, Evidence::Native(_)))
        .map(|(r, _)| format!("{:?}", r.role))
        .collect();
    match verify_bundle(&backend, &release, &req.as_request(), &reply, &children) {
        Ok(s) => {
            println!(
                "ACCEPT-UNPROVEN request {} reply {:?} root {} end {:?}",
                hex(&s.request_id),
                reply,
                hex(&s.reply_root),
                s.end
            );
            println!(
                "evidence: {} emulator-executed layer children (no proofs); native host claims: {native:?}",
                children.len() - native.len()
            );
            Ok(())
        }
        Err(e) => {
            println!("REJECT {e:?}");
            std::process::exit(2)
        }
    }
}

/// Run forged records for one layer through the emulator and the verifier.
fn forge(o: &Opts) -> Res<()> {
    let out = o.path("out")?;
    std::fs::create_dir_all(&out)?;
    let base = o.path("bundle")?;
    let pins = Pins::load(&o.path("inventory")?)?;
    let ck = Checkpoint::open(&o.path("model")?)?;
    let elf = o.path("elf")?;
    let elf_sha = sha(&std::fs::read(&elf)?);
    let release = pins.release(o)?;
    let req = Req::from(o)?;
    let q = req.as_request();
    let model = release.model();
    let rq = request_id(&release, &q);
    let layer: u32 = o.get("layer")?.parse()?;
    let emu = Emu {
        elf,
        ziskemu: o.path("ziskemu")?,
        args: o
            .0
            .get("emu-args")
            .map(|s| s.split_whitespace().map(String::from).collect())
            .unwrap_or_default(),
    };
    let (reply, honest) = load_bundle(&base)?;
    // Rebuild the honest input of this layer from the bundle's own chain.
    let consumed_t = honest[0].0.positions;
    let feedback = &reply[..reply.len() - 1];
    let consumed: Vec<u32> = req.prompt.iter().chain(feedback).copied().collect();
    assert_eq!(consumed.len() as u32, consumed_t);
    let predictions = reply.len() as u32;
    let tied = ck.raw("model.embed_tokens.weight")?;
    let mut x = embed(&tied, &consumed)?;
    drop(tied);
    for i in 0..layer {
        let rec = encode(model, release.suite, rq, i, predictions, &x, &layer_tensors(&ck, i)?);
        x = execute(&rec, sha).map_err(|e| e.to_string())?.output;
    }
    let tensors = layer_tensors(&ck, layer)?;
    let honest_record = encode(model, release.suite, rq, layer, predictions, &x, &tensors);
    let honest_receipt = execute(&honest_record, sha)
        .map_err(|e| e.to_string())?
        .receipt;
    if honest_receipt.input != honest[layer as usize + 1].0.input {
        return Err("rebuilt input does not match the bundle".into());
    }
    let prefix = pci_layer::FIXED + x.len() * 4;
    let offset = |slot: usize| prefix + TENSOR_LEN[..slot].iter().map(|n| 2 * n).sum::<usize>();
    let mut cases: Vec<(&str, Vec<u8>)> = Vec::new();
    let mut flip = |name: &'static str, at: usize, mask: u8| {
        let mut r = honest_record.clone();
        r[at] ^= mask;
        cases.push((name, r));
    };
    // Full executions: the guest accepts and binds the change.
    flip("weight-bit-q_proj", offset(9) + 2 * 1234 + 1, 0x01);
    flip("scalar-bit-k_norm", offset(5), 0x01);
    flip("request-id", 72, 0x01);
    flip("activation-bit", pci_layer::FIXED + 4 * 17, 0x01);
    flip("layer-index-swap", 104, 0x01);
    // Rejected while parsing.
    flip("flag-dotf8", 112 + 40, 0x04);
    flip("flag-hash-input", 112 + 40, 0x01);
    flip("eps-bit", 112 + 36, 0x01);
    flip("positions", 108, 0x01);
    flip("predictions-over-positions", 112 + 44 + 3, 0x80);
    let mut trailing = honest_record.clone();
    trailing.push(0);
    cases.push(("trailing-byte", trailing));
    let mut report = Vec::new();
    let started: Vec<_> = cases
        .into_iter()
        .map(|(name, record)| Ok((name, emu.start(&out, name, &record, false)?, record)))
        .collect::<Res<_>>()?;
    for (name, job, record) in started {
        let (ok, steps, file) = Emu::finish(job)?;
        let mut verdict = "guest-rejected (no public output)".to_string();
        if ok {
            // Substitute this child's evidence and its honestly computed
            // receipt (what the guest really bound) into the honest bundle.
            let forged = execute(&record, sha).map_err(|e| e.to_string())?.receipt;
            let digest: Hash = std::fs::read(&file)?[..32].try_into()?;
            let matches = digest == pci_layer::digest_of(&record, &forged);
            let mut children: Vec<_> = load_bundle(&base)?.1;
            let slot = layer as usize + 1;
            let mut verdicts = vec![];
            for (label, receipt) in [
                ("guest-receipt", forged),
                ("honest-receipt", honest_receipt.clone()),
            ] {
                children[slot] = (receipt, Evidence::Emulated(file.clone(), elf_sha));
                let r = verify_bundle(
                    &EmulatorTranscript {
                        allow_native: true,
                        program: elf_sha,
                    },
                    &release,
                    &q,
                    &reply,
                    &children,
                );
                verdicts.push(format!(
                    "{label}: {:?}",
                    r.err().map_or("ACCEPT".into(), |e| format!("{e:?}"))
                ));
            }
            verdict = format!(
                "guest ran, digest matches native forged receipt: {matches}; verifier {verdicts:?}"
            );
        }
        println!("{name}: steps {steps}; {verdict}");
        report.push(json!({"case": name, "steps": steps, "verdict": verdict}));
    }
    std::fs::write(out.join("forge.json"), serde_json::to_vec_pretty(&report)?)?;
    Ok(())
}

fn main() -> Res<()> {
    let (cmd, o) = Opts::parse()?;
    match cmd.as_str() {
        "run" => run(&o),
        "verify" => verify(&o),
        "forge" => forge(&o),
        _ => Err("command: run | verify | forge".into()),
    }
}
