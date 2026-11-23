//! Client side of the token-ID greedy Qwen3-4B suite: release pins, request,
//! and the fail-closed verifier of a bundle of child proofs
//! (embedding, layers 0..35, every head shard).
//!
//! Trust: the client holds a release file it obtained independently (inventory
//! digests, row-commitment root it derived itself, setup key and approved
//! program VKs per role) and its own request. Everything in a bundle is
//! untrusted until its proof verifies under the pinned setup key and an
//! approved program VK, and the typed receipt it carries hashes to the proof's
//! public digest.
use pci_receipts::{
    Hash, LAYERS, Receipt, Release, Request, Role, STAGES, VOCAB, consumed_root, reply_root,
    request_id, stage_tensors, weights_root,
};
use pci_stages::head::{self, ShardReceipt};
use serde_json::{Map, Value, json};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::path::Path;

pub type Res<T> = Result<T, String>;

pub fn sha(b: &[u8]) -> Hash {
    Sha256::digest(b).into()
}

pub fn hex(h: &[u8]) -> String {
    h.iter().map(|b| format!("{b:02x}")).collect()
}

pub fn unhex(s: &str) -> Res<Hash> {
    if s.len() != 64 || !s.bytes().all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase()) {
        return Err(format!("not a lowercase 32-byte hex value: {s:?}"));
    }
    let mut h = [0u8; 32];
    for (i, b) in h.iter_mut().enumerate() {
        *b = u8::from_str_radix(&s[2 * i..2 * i + 2], 16).unwrap();
    }
    Ok(h)
}

fn read_json(path: &Path) -> Res<Value> {
    let b = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    serde_json::from_slice(&b).map_err(|e| format!("{}: {e}", path.display()))
}

/// Strict object access: exactly these keys.
fn object<'a>(v: &'a Value, keys: &[&str], what: &str) -> Res<&'a Map<String, Value>> {
    let o = v.as_object().ok_or(format!("{what}: not an object"))?;
    let mut have: Vec<&str> = o.keys().map(String::as_str).collect();
    let mut want = keys.to_vec();
    have.sort();
    want.sort();
    if have != want {
        return Err(format!("{what}: keys {have:?}, expected {want:?}"));
    }
    Ok(o)
}

fn get_hash(o: &Map<String, Value>, k: &str) -> Res<Hash> {
    unhex(o[k].as_str().ok_or(format!("{k}: not a string"))?)
}

fn get_u32(o: &Map<String, Value>, k: &str) -> Res<u32> {
    o[k].as_u64()
        .and_then(|v| u32::try_from(v).ok())
        .ok_or(format!("{k}: not a u32"))
}

fn get_u32s(v: &Value, what: &str) -> Res<Vec<u32>> {
    v.as_array()
        .ok_or(format!("{what}: not an array"))?
        .iter()
        .map(|x| x.as_u64().and_then(|v| u32::try_from(v).ok()).ok_or(format!("{what}: not u32")))
        .collect()
}

fn get_vk(v: &Value, what: &str) -> Res<[u64; 4]> {
    let a = v.as_array().ok_or(format!("{what}: not an array"))?;
    if a.len() != 4 {
        return Err(format!("{what}: a key is 4 u64 words"));
    }
    let mut out = [0u64; 4];
    for (o, x) in out.iter_mut().zip(a) {
        *o = x.as_u64().ok_or(format!("{what}: not u64"))?;
    }
    Ok(out)
}

// ------------------------------------------------------------------ inventory

/// The enrollment inventory of the publisher checkpoint (enroll.py v1).
pub struct Inventory {
    pub checkpoint: Hash,
    pub tensors: HashMap<String, (Hash, Vec<u64>)>,
    pub file_sha256: Hash,
}

impl Inventory {
    /// Recomputes the checkpoint ID from the inventory body and checks the stage
    /// roster, as `pci-layer-host` does.
    pub fn load(path: &Path) -> Res<Self> {
        let raw = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
        let v: Value = serde_json::from_slice(&raw).map_err(|e| e.to_string())?;
        let checkpoint = unhex(v["checkpoint_id"].as_str().ok_or("checkpoint_id")?)?;
        let mut body = v.as_object().ok_or("inventory")?.clone();
        body.remove("checkpoint_id");
        let mut pre = b"PCI/checkpoint-inventory/v1\0".to_vec();
        pre.extend(serde_json::to_vec(&body).map_err(|e| e.to_string())?);
        if sha(&pre) != checkpoint {
            return Err("checkpoint_id does not match the inventory contents".into());
        }
        let mut tensors = HashMap::new();
        for (name, t) in v["tensors"].as_object().ok_or("tensors")? {
            if t["dtype"] != "BF16" {
                return Err(format!("{name}: not BF16"));
            }
            let shape = t["shape"]
                .as_array()
                .ok_or("shape")?
                .iter()
                .map(|d| d.as_u64().ok_or("shape"))
                .collect::<Result<Vec<_>, _>>()?;
            tensors.insert(name.clone(), (unhex(t["sha256"].as_str().ok_or("sha256")?)?, shape));
        }
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
            if listed != stage_tensors(roster_role(i)) {
                return Err(format!("inventory stage {i} roster differs"));
            }
        }
        Ok(Inventory {
            checkpoint,
            tensors,
            file_sha256: sha(&raw),
        })
    }

    pub fn digest(&self, name: &str) -> Res<Hash> {
        self.tensors.get(name).map(|t| t.0).ok_or(format!("unpinned {name}"))
    }

    pub fn stage_weights(&self) -> Res<Vec<Hash>> {
        (0..STAGES)
            .map(|i| {
                let r = roster_role(i);
                let d = stage_tensors(r)
                    .iter()
                    .map(|n| self.digest(n))
                    .collect::<Res<Vec<_>>>()?;
                Ok(weights_root(r, &d))
            })
            .collect()
    }
}

pub fn roster_role(i: usize) -> Role {
    match i {
        0 => Role::Embed,
        1..=LAYERS => Role::Layer((i - 1) as u32),
        _ => Role::Head,
    }
}

// ------------------------------------------------------------------ checkpoint

/// Raw tensor bytes from the safetensors shards named by the index.
pub struct Checkpoint {
    dir: std::path::PathBuf,
    map: HashMap<String, String>,
}

impl Checkpoint {
    pub fn open(dir: &Path) -> Res<Self> {
        let idx = read_json(&dir.join("model.safetensors.index.json"))?;
        let map = idx["weight_map"]
            .as_object()
            .ok_or("weight_map")?
            .iter()
            .map(|(k, v)| (k.clone(), v.as_str().unwrap_or("").to_string()))
            .collect();
        Ok(Checkpoint {
            dir: dir.to_path_buf(),
            map,
        })
    }

    pub fn raw(&self, name: &str) -> Res<Vec<u8>> {
        use std::io::{Read, Seek, SeekFrom};
        let file = self.map.get(name).ok_or(format!("{name}: not in index"))?;
        let mut f = std::fs::File::open(self.dir.join(file)).map_err(|e| e.to_string())?;
        let mut len = [0u8; 8];
        f.read_exact(&mut len).map_err(|e| e.to_string())?;
        let mut header = vec![0u8; u64::from_le_bytes(len) as usize];
        f.read_exact(&mut header).map_err(|e| e.to_string())?;
        let header: Value = serde_json::from_slice(&header).map_err(|e| e.to_string())?;
        let t = &header[name];
        if t["dtype"] != "BF16" {
            return Err(format!("{name}: not BF16"));
        }
        let o = t["data_offsets"].as_array().filter(|o| o.len() == 2).ok_or("data_offsets")?;
        let (a, b) = (o[0].as_u64().ok_or("offset")?, o[1].as_u64().ok_or("offset")?);
        if b < a {
            return Err("data_offsets".into());
        }
        let mut out = vec![0u8; (b - a) as usize];
        f.seek(SeekFrom::Start(8 + u64::from_le_bytes(len) + a)).map_err(|e| e.to_string())?;
        f.read_exact(&mut out).map_err(|e| e.to_string())?;
        Ok(out)
    }

    /// The tensor bytes, after checking them against the pinned inventory digest.
    pub fn authenticated(&self, inv: &Inventory, name: &str) -> Res<Vec<u8>> {
        let b = self.raw(name)?;
        if sha(&b) != inv.digest(name)? {
            return Err(format!("{name}: bytes differ from the pinned inventory digest"));
        }
        Ok(b)
    }
}

pub const TIED: &str = "model.embed_tokens.weight";
pub const NORM: &str = "model.norm.weight";

// ------------------------------------------------------------------ release

pub use pci_receipts::suite_id;

/// Client-held release pins. Loaded from a file the client obtained
/// independently of any bundle.
#[derive(Clone, Debug)]
pub struct Pins {
    pub checkpoint: Hash,
    pub stage_weights: Vec<Hash>,
    pub rows_root: Hash,
    pub norm_sha256: Hash,
    pub shard_log2: u32,
    pub context: u32,
    pub stop_ids: Vec<u32>,
    /// vadcop_final leaf setup key of the proving key.
    pub setup_vk: [u64; 4],
    pub embed_programs: Vec<[u64; 4]>,
    pub layer_programs: Vec<[u64; 4]>,
    pub head_programs: Vec<[u64; 4]>,
}

impl Pins {
    pub fn suite(&self) -> Hash {
        suite_id(self.shard_log2)
    }
    pub fn release(&self) -> Release {
        Release {
            checkpoint: self.checkpoint,
            weights: self.stage_weights.clone(),
            suite: self.suite(),
            context: self.context,
            stop_ids: self.stop_ids.clone(),
        }
    }
    pub fn embed_weights(&self) -> Hash {
        pci_stages::embed_weights(sha, &self.rows_root)
    }
    pub fn head_weights(&self) -> Hash {
        pci_stages::head_weights(sha, &self.norm_sha256, &self.rows_root)
    }

    pub fn to_json(&self) -> Value {
        let h = |v: &[Hash]| v.iter().map(|x| hex(x)).collect::<Vec<_>>();
        json!({
            "format": "pci-client-release/v1",
            "checkpoint": hex(&self.checkpoint),
            "stage_weights": h(&self.stage_weights),
            "rows_root": hex(&self.rows_root),
            "norm_sha256": hex(&self.norm_sha256),
            "head_shard_log2": self.shard_log2,
            "context": self.context,
            "stop_ids": self.stop_ids,
            "setup_vk": self.setup_vk,
            "programs": {
                "embed": self.embed_programs,
                "layer": self.layer_programs,
                "head": self.head_programs,
            },
            "derived": {
                "suite": hex(&self.suite()),
                "model": hex(&self.release().model()),
                "embed_weights": hex(&self.embed_weights()),
                "head_weights": hex(&self.head_weights()),
            },
        })
    }

    pub fn from_json(v: &Value) -> Res<Self> {
        let o = object(
            v,
            &[
                "format", "checkpoint", "stage_weights", "rows_root", "norm_sha256",
                "head_shard_log2", "context", "stop_ids", "setup_vk", "programs", "derived",
            ],
            "release",
        )?;
        if o["format"] != "pci-client-release/v1" {
            return Err("release format".into());
        }
        let stage_weights = o["stage_weights"]
            .as_array()
            .ok_or("stage_weights")?
            .iter()
            .map(|x| unhex(x.as_str().unwrap_or("")))
            .collect::<Res<Vec<_>>>()?;
        let p = object(&o["programs"], &["embed", "layer", "head"], "programs")?;
        let vks = |k: &str| -> Res<Vec<[u64; 4]>> {
            p[k].as_array()
                .ok_or(format!("programs.{k}"))?
                .iter()
                .map(|x| get_vk(x, k))
                .collect()
        };
        let pins = Pins {
            checkpoint: get_hash(o, "checkpoint")?,
            stage_weights,
            rows_root: get_hash(o, "rows_root")?,
            norm_sha256: get_hash(o, "norm_sha256")?,
            shard_log2: get_u32(o, "head_shard_log2")?,
            context: get_u32(o, "context")?,
            stop_ids: get_u32s(&o["stop_ids"], "stop_ids")?,
            setup_vk: get_vk(&o["setup_vk"], "setup_vk")?,
            embed_programs: vks("embed")?,
            layer_programs: vks("layer")?,
            head_programs: vks("head")?,
        };
        // The derived block is informational; it must agree with the pins.
        if pins.to_json()["derived"] != o["derived"] {
            return Err("release: derived values disagree with the pins".into());
        }
        Ok(pins)
    }
}

pub struct ClientRequest {
    pub nonce: Hash,
    pub prompt: Vec<u32>,
    pub max_reply: u32,
}

impl ClientRequest {
    pub fn from_json(v: &Value) -> Res<Self> {
        let o = object(v, &["format", "nonce", "prompt", "max_reply"], "request")?;
        if o["format"] != "pci-request/v1" {
            return Err("request format".into());
        }
        Ok(ClientRequest {
            nonce: get_hash(o, "nonce")?,
            prompt: get_u32s(&o["prompt"], "prompt")?,
            max_reply: get_u32(o, "max_reply")?,
        })
    }
    pub fn to_json(&self) -> Value {
        json!({"format": "pci-request/v1", "nonce": hex(&self.nonce), "prompt": self.prompt, "max_reply": self.max_reply})
    }
    pub fn as_request(&self) -> Request<'_> {
        Request {
            nonce: self.nonce,
            prompt: &self.prompt,
            max_reply: self.max_reply,
        }
    }
}

// ------------------------------------------------------------------ proofs

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    Embed,
    Layer,
    Head,
}

/// Verifies one child proof and returns its authenticated 32-byte digest.
pub trait ChildVerifier {
    fn verify(&self, kind: Kind, proof: &[u8]) -> Res<Hash>;
}

/// The real ZisK backend: `zisk-verifier` (generated from the pinned proving
/// key) under the pinned setup key, the role's approved program VKs, a leaf
/// proof, and publics = 8 little-endian u32 words of the digest, then zeros.
pub struct Zisk<'a> {
    pub pins: &'a Pins,
}

/// What a verified ZisK leaf proof states.
#[derive(Debug)]
pub struct ProofFacts {
    pub program_vk: [u64; 4],
    pub digest: Hash,
}

pub fn verify_zisk_leaf(
    proof: &[u8],
    setup_vk: &[u64; 4],
    approved: &[[u64; 4]],
) -> Res<ProofFacts> {
    let saved = zisk_verifier::decode_saved(proof).map_err(|e| format!("malformed proof: {e}"))?;
    if saved.proof.public_values.len() != zisk_verifier::expected_n_publics(saved.proof.compressed) {
        return Err("proof publics have the wrong width".into());
    }
    let claimed = &zisk_verifier::program_publics(&saved.proof.public_values)[..4];
    let program: [u64; 4] = claimed.try_into().unwrap();
    // Choose among approved programs by the proof's claim; the verifier then
    // checks the claim is what the STARK commits to.
    let expected = approved
        .iter()
        .find(|vk| **vk == program)
        .ok_or_else(|| format!("program VK {program:?} is not approved for this role"))?;
    let v = zisk_verifier::verify(&saved.proof, setup_vk).map_err(|e| format!("proof rejected: {e}"))?;
    if v.program_vk != *expected {
        return Err("verified program VK differs from the approved one".into());
    }
    if v.is_aggregate != Some(false) {
        return Err("not a leaf proof".into());
    }
    let out = v.outputs_u32().ok_or("outputs are not u32")?;
    if out[8..].iter().any(|w| *w != 0) {
        return Err("public outputs beyond the 32-byte digest are not zero".into());
    }
    let mut digest = [0u8; 32];
    for (d, w) in digest.chunks_exact_mut(4).zip(&out[..8]) {
        d.copy_from_slice(&w.to_le_bytes());
    }
    Ok(ProofFacts {
        program_vk: v.program_vk,
        digest,
    })
}

impl ChildVerifier for Zisk<'_> {
    fn verify(&self, kind: Kind, proof: &[u8]) -> Res<Hash> {
        let approved = match kind {
            Kind::Embed => &self.pins.embed_programs,
            Kind::Layer => &self.pins.layer_programs,
            Kind::Head => &self.pins.head_programs,
        };
        verify_zisk_leaf(proof, &self.pins.setup_vk, approved).map(|f| f.digest)
    }
}

// ------------------------------------------------------------------ bundle

pub enum ChildReceipt {
    Stage(Receipt),
    Shard(ShardReceipt),
}

pub struct Child {
    pub receipt: ChildReceipt,
    pub proof: Vec<u8>,
    pub proof_file: String,
}

pub struct Bundle {
    pub reply: Vec<u32>,
    pub children: Vec<Child>,
}

pub fn receipt_to_json(r: &Receipt) -> Value {
    let (kind, index) = match r.role {
        Role::Embed => ("embed", 0),
        Role::Layer(i) => ("layer", i),
        Role::Head => ("head", 0),
    };
    json!({
        "kind": kind, "index": index, "positions": r.positions,
        "first_prediction": r.first_prediction, "predictions": r.predictions,
        "input": hex(&r.input), "output": hex(&r.output), "weights": hex(&r.weights),
    })
}

pub fn shard_to_json(r: &ShardReceipt) -> Value {
    json!({
        "kind": "head-shard", "shard_log2": r.shard_log2, "shard": r.shard,
        "positions": r.positions, "first_prediction": r.first_prediction,
        "predictions": r.predictions, "input": hex(&r.input), "weights": hex(&r.weights),
        "best": r.best.iter().map(|(id, bits)| json!([id, bits])).collect::<Vec<_>>(),
    })
}

fn receipt_from_json(v: &Value) -> Res<ChildReceipt> {
    let kind = v["kind"].as_str().ok_or("receipt kind")?;
    if kind == "head-shard" {
        let o = object(
            v,
            &["kind", "shard_log2", "shard", "positions", "first_prediction", "predictions", "input", "weights", "best"],
            "shard receipt",
        )?;
        let best = o["best"]
            .as_array()
            .ok_or("best")?
            .iter()
            .map(|p| {
                let p = get_u32s(p, "best")?;
                if p.len() != 2 {
                    return Err("best entry".to_string());
                }
                Ok((p[0], p[1]))
            })
            .collect::<Res<Vec<_>>>()?;
        return Ok(ChildReceipt::Shard(ShardReceipt {
            shard_log2: get_u32(o, "shard_log2")?,
            shard: get_u32(o, "shard")?,
            positions: get_u32(o, "positions")?,
            first_prediction: get_u32(o, "first_prediction")?,
            predictions: get_u32(o, "predictions")?,
            input: get_hash(o, "input")?,
            weights: get_hash(o, "weights")?,
            best,
        }));
    }
    let o = object(
        v,
        &["kind", "index", "positions", "first_prediction", "predictions", "input", "output", "weights"],
        "receipt",
    )?;
    let role = match (kind, get_u32(o, "index")?) {
        ("embed", 0) => Role::Embed,
        ("layer", i) if (i as usize) < LAYERS => Role::Layer(i),
        _ => return Err(format!("receipt kind {kind:?} / index")),
    };
    Ok(ChildReceipt::Stage(Receipt {
        role,
        positions: get_u32(o, "positions")?,
        first_prediction: get_u32(o, "first_prediction")?,
        predictions: get_u32(o, "predictions")?,
        input: get_hash(o, "input")?,
        output: get_hash(o, "output")?,
        weights: get_hash(o, "weights")?,
    }))
}

impl Bundle {
    /// Loads `bundle.json` and every proof it names from `dir`. Proof names
    /// must be plain file names inside the bundle directory.
    pub fn load(dir: &Path) -> Res<Self> {
        let v = read_json(&dir.join("bundle.json"))?;
        let o = object(&v, &["format", "reply", "children"], "bundle")?;
        if o["format"] != "pci-proof-bundle/v1" {
            return Err("bundle format".into());
        }
        let reply = get_u32s(&o["reply"], "reply")?;
        let mut children = vec![];
        for c in o["children"].as_array().ok_or("children")? {
            let co = object(c, &["receipt", "proof"], "child")?;
            // `null`: no per-child proof (compact format). Bundle verification
            // then rejects that child, since empty bytes never verify.
            let (name, proof) = if co["proof"].is_null() {
                ("", vec![])
            } else {
                let name = co["proof"].as_str().ok_or("proof name")?;
                if name.is_empty() || name.contains('/') || name.starts_with('.') {
                    return Err(format!("proof name {name:?} must be a plain file name"));
                }
                (name, std::fs::read(dir.join(name)).map_err(|e| format!("{name}: {e}"))?)
            };
            children.push(Child {
                receipt: receipt_from_json(&co["receipt"])?,
                proof,
                proof_file: name.to_string(),
            });
        }
        Ok(Bundle { reply, children })
    }

    pub fn to_json(reply: &[u32], children: &[(Value, String)]) -> Value {
        json!({
            "format": "pci-proof-bundle/v1",
            "reply": reply,
            "children": children.iter().map(|(r, p)| json!({"receipt": r, "proof": p})).collect::<Vec<_>>(),
        })
    }
}

#[derive(Debug, PartialEq, Eq)]
pub enum Reject {
    Policy,
    Length,
    Token,
    Termination,
    Roster(String),
    Proof(usize, String),
    Receipt(usize, String),
    Weights(usize),
    Prediction,
}

#[derive(Debug, PartialEq, Eq)]
pub struct Statement {
    pub model: Hash,
    pub suite: Hash,
    pub request_id: Hash,
    pub reply_root: Hash,
    pub stopped: bool,
    pub predicted: Vec<u32>,
}

/// The request-level checks: policy, lengths, IDs, first-stop and limit rule.
/// Returns (P, R, T, stopped).
pub fn check_request(pins: &Pins, req: &ClientRequest, reply: &[u32]) -> Result<(u32, u32, u32, bool), Reject> {
    if !(1..=40_960).contains(&pins.context)
        || pins.stage_weights.len() != STAGES
        || pins.stop_ids.iter().any(|id| *id >= VOCAB)
        || !pins.stop_ids.windows(2).all(|w| w[0] < w[1])
        || !(head::MIN_LOG2..=head::MAX_LOG2).contains(&pins.shard_log2)
    {
        return Err(Reject::Policy);
    }
    let p = u32::try_from(req.prompt.len()).map_err(|_| Reject::Length)?;
    let r = u32::try_from(reply.len()).map_err(|_| Reject::Length)?;
    if p == 0 || r == 0 || r > req.max_reply || p.checked_add(req.max_reply).is_none_or(|v| v > pins.context) {
        return Err(Reject::Length);
    }
    if req.prompt.iter().chain(reply).any(|id| *id >= VOCAB) {
        return Err(Reject::Token);
    }
    let stop = |id: &u32| pins.stop_ids.binary_search(id).is_ok();
    let (last, feedback) = reply.split_last().unwrap();
    if feedback.iter().any(stop) {
        return Err(Reject::Termination);
    }
    let stopped = stop(last);
    if !stopped && r != req.max_reply {
        return Err(Reject::Termination);
    }
    Ok((p, r, p + r - 1, stopped))
}

/// The join shared by bundle and compact verification: request rules, roster,
/// weights, links, windows and the greedy full-vocabulary prediction.
/// `authenticate(i, kind, digest)` runs for each child, in order, right after
/// its receipt digest is computed and before any other field of it is used;
/// it must reject unless that digest is proven for that role.
pub fn join(
    pins: &Pins,
    req: &ClientRequest,
    reply: &[u32],
    receipts: &[&ChildReceipt],
    authenticate: &mut dyn FnMut(usize, Kind, &Hash) -> Result<(), Reject>,
) -> Result<Statement, Reject> {
    let (p, r, t, stopped) = check_request(pins, req, reply)?;
    let shards = head::shard_count(pins.shard_log2) as usize;
    if receipts.len() != 1 + LAYERS + shards {
        return Err(Reject::Roster(format!(
            "{} children, the roster is 1 + {LAYERS} + {shards}",
            receipts.len()
        )));
    }
    let release = pins.release();
    let model = release.model();
    let suite = release.suite;
    let request = request_id(&release, &req.as_request());
    let feedback = &reply[..reply.len() - 1];
    let mut previous = consumed_root(&req.prompt, feedback);
    let mut shard_receipts = Vec::with_capacity(shards);
    for (i, receipt) in receipts.iter().enumerate() {
        let (kind, digest) = match (*receipt, i) {
            (ChildReceipt::Stage(rc), 0) if rc.role == Role::Embed => {
                if rc.weights != pins.embed_weights() {
                    return Err(Reject::Weights(i));
                }
                (Kind::Embed, rc.digest(model, suite, request))
            }
            (ChildReceipt::Stage(rc), 1..=LAYERS) if rc.role == Role::Layer((i - 1) as u32) => {
                if rc.weights != pins.stage_weights[i] {
                    return Err(Reject::Weights(i));
                }
                (Kind::Layer, rc.digest(model, suite, request))
            }
            (ChildReceipt::Shard(s), _) if i > LAYERS && s.shard as usize == i - 1 - LAYERS => {
                if s.weights != pins.head_weights() {
                    return Err(Reject::Weights(i));
                }
                (Kind::Head, s.digest(sha, model, suite, request))
            }
            _ => return Err(Reject::Roster(format!("child {i} has the wrong role or order"))),
        };
        // The proof first: nothing from the receipt counts unless it is proven.
        authenticate(i, kind, &digest)?;
        match *receipt {
            ChildReceipt::Stage(rc) => {
                let window = if rc.role == Role::Embed { (0, 0) } else { (p - 1, r) };
                if rc.positions != t || (rc.first_prediction, rc.predictions) != window {
                    return Err(Reject::Receipt(i, "positions or window".into()));
                }
                if rc.input != previous {
                    return Err(Reject::Receipt(i, "input does not link to the previous output".into()));
                }
                previous = rc.output;
            }
            ChildReceipt::Shard(s) => {
                let lo = s.shard << pins.shard_log2;
                let hi = lo + head::shard_rows(pins.shard_log2, s.shard);
                if s.shard_log2 != pins.shard_log2
                    || s.positions != t
                    || (s.first_prediction, s.predictions) != (p - 1, r)
                    || s.best.len() != r as usize
                    || s.best.iter().any(|(id, _)| !(lo..hi).contains(id))
                {
                    return Err(Reject::Receipt(i, "shard shape, window or winner range".into()));
                }
                if s.input != previous {
                    return Err(Reject::Receipt(i, "shard input is not the layer-35 output".into()));
                }
                shard_receipts.push(s.clone());
            }
        }
    }
    let predicted = head::combine(&shard_receipts).map_err(|_| Reject::Prediction)?;
    if predicted != reply {
        return Err(Reject::Prediction);
    }
    Ok(Statement {
        model,
        suite,
        request_id: request,
        reply_root: reply_root(reply),
        stopped,
        predicted,
    })
}

/// Fail-closed bundle verification: every child's ZisK proof is verified under
/// the pinned setup key and its role's approved program, and its public digest
/// must equal the digest of the typed receipt the bundle carries. Any missing,
/// extra, reordered, unproven, mislinked or mispredicting child rejects.
pub fn verify_bundle(
    backend: &dyn ChildVerifier,
    pins: &Pins,
    req: &ClientRequest,
    bundle: &Bundle,
) -> Result<Statement, Reject> {
    let receipts: Vec<&ChildReceipt> = bundle.children.iter().map(|c| &c.receipt).collect();
    join(pins, req, &bundle.reply, &receipts, &mut |i, kind, digest| {
        let proven = backend
            .verify(kind, &bundle.children[i].proof)
            .map_err(|e| Reject::Proof(i, e))?;
        if proven != *digest {
            return Err(Reject::Receipt(i, "receipt does not hash to the proven digest".into()));
        }
        Ok(())
    })
}

// ------------------------------------------------------------------ compact

/// Ordered child-digest accumulator the recursive join commits to
/// (`harness/client/recursion/`): over Goldilocks with the BLAKE3 linear hash
/// of the blake3 proving-key family (`proofman_fields::Blake3_8`, the hash the
/// circom `LinearHash` gadget computes).
///
/// ```text
/// leaf(d)      = (1, H(d as 8 u32 limbs, little-endian))
/// node(a, b)   = (a.n + b.n, H(a.n, a.acc[4], b.n, b.acc[4]))
/// fold(lo, hi) = leaf(d[lo])                                   if hi - lo == 1
///              = node(fold(lo, m), fold(m, hi)), m = lo + 2^(ceil(log2(hi-lo)) - 1)
/// ```
/// Counts make every node bind its subtree size; the client accepts only this
/// canonical shape (an honest prover folds exactly so).
pub mod accumulator {
    use super::Hash;
    use proofman_fields::{Blake3_8, Goldilocks, Hash as FieldHash, PrimeField64};

    pub type Acc = (u64, [u64; 4]);

    fn h(words: &[u64]) -> [u64; 4] {
        let e: Vec<Goldilocks> = words.iter().map(|w| Goldilocks::from_u64(*w)).collect();
        let out = <Blake3_8 as FieldHash<Goldilocks>>::linear_hash(&e);
        core::array::from_fn(|i| out[i].as_canonical_u64())
    }

    pub fn leaf(d: &Hash) -> Acc {
        let limbs: Vec<u64> = d.chunks_exact(4).map(|c| u32::from_le_bytes(c.try_into().unwrap()) as u64).collect();
        (1, h(&limbs))
    }

    pub fn node(a: &Acc, b: &Acc) -> Acc {
        let mut w = vec![a.0];
        w.extend(a.1);
        w.push(b.0);
        w.extend(b.1);
        (a.0 + b.0, h(&w))
    }

    pub fn fold(d: &[Hash]) -> Acc {
        assert!(!d.is_empty());
        if d.len() == 1 {
            return leaf(&d[0]);
        }
        let m = d.len().next_power_of_two() / 2;
        node(&fold(&d[..m]), &fold(&d[m..]))
    }
}

/// Compact verification: one aggregate proof from the pinned recurser over
/// all children, plus the typed receipts (no child proofs). The recurser's
/// allow-list admits only the approved embed/layer/head programs, and each
/// program can only emit digests of its own role (domain-separated receipt
/// encodings), so the aggregate's accumulator over the ordered digests stands
/// for one verified child proof per roster slot.
pub fn verify_compact(
    pins: &Pins,
    recurser_vk: &[u64; 4],
    req: &ClientRequest,
    reply: &[u32],
    receipts: &[ChildReceipt],
    aggregate_proof: &[u8],
) -> Result<Statement, Reject> {
    let saved = zisk_verifier::decode_saved(aggregate_proof)
        .map_err(|e| Reject::Proof(usize::MAX, format!("malformed aggregate: {e}")))?;
    let v = zisk_verifier::verify(&saved.proof, recurser_vk)
        .map_err(|e| Reject::Proof(usize::MAX, format!("aggregate rejected: {e}")))?;
    if v.is_aggregate != Some(true) || v.program_vk != *recurser_vk {
        return Err(Reject::Proof(usize::MAX, "not an aggregate of the pinned recurser".into()));
    }
    let refs: Vec<&ChildReceipt> = receipts.iter().collect();
    let mut digests = Vec::with_capacity(receipts.len());
    let s = join(pins, req, reply, &refs, &mut |_, _, d| {
        digests.push(*d);
        Ok(())
    })?;
    let (n, acc) = accumulator::fold(&digests);
    if v.outputs[0] != n || v.outputs[1..5] != acc || v.outputs[5..].iter().any(|w| *w != 0) {
        return Err(Reject::Receipt(usize::MAX, "aggregate accumulator differs from the receipts".into()));
    }
    Ok(s)
}
