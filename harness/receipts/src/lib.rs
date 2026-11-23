//! Client-side join for the token-ID, BF16, greedy Qwen3-4B suite.
//! The backend adapter owns key selection and proof verification. Each guest
//! commits a 32-byte receipt digest followed by the zkVM's zero padding.
use sha2::{Digest, Sha256};

pub const SUITE_TEXT: &str = "qwen3-4b token-ids greedy-lowest-id; layer-record v2; child-receipt v3; \
mac-arm64-bf16-sdpa-v1; torch cf30153c; transformers 5.15.1; libm 0.2.16; rows-v1 depth 18; embed-record v1; \
head-shard-record v1 receipt v1; zisk leaf proofs, publics = 32-byte digest then zeros; bundle v1";

/// SuiteID: the function and encodings. Approved programs are pinned per role
/// in the release, so kernels computing the same function share digests.
pub fn suite_id(shard_log2: u32) -> Hash {
    let mut pre = b"PCI/v1/suite\0".to_vec();
    pre.extend(SUITE_TEXT.as_bytes());
    pre.extend(b"; head shard log2 ");
    pre.extend(shard_log2.to_string().as_bytes());
    Sha256::digest(&pre).into()
}

pub type Hash = [u8; 32];
pub const VOCAB: u32 = 151936;
pub const LAYERS: usize = 36;
pub const DIM: u32 = 2560;
/// Embedding, 36 layers, head.
pub const STAGES: usize = LAYERS + 2;
pub const LAYER_TENSORS: [&str; 11] = [
    "input_layernorm",
    "mlp.down_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "post_attention_layernorm",
    "self_attn.k_norm",
    "self_attn.k_proj",
    "self_attn.o_proj",
    "self_attn.q_norm",
    "self_attn.q_proj",
    "self_attn.v_proj",
];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Role {
    Embed,
    Layer(u32),
    Head,
}

/// Trusted client configuration loaded independently of the bundle. The
/// stage weight roots come from the client's own enrollment of the pinned
/// publisher checkpoint, never from the prover. ModelID is derived from them.
pub struct Release {
    pub checkpoint: Hash,
    /// One root per stage in roster order: embedding, layers 0..35, head.
    pub weights: Vec<Hash>,
    pub suite: Hash,
    pub context: u32,
    pub stop_ids: Vec<u32>,
}

impl Release {
    pub fn model(&self) -> Hash {
        model_id(&self.checkpoint, &self.weights)
    }
}

/// The 64-byte segment header the pinned BF16 serving profile accepts,
/// in `qwen_common::Header` layout: kind 1, dim, hidden, heads, kv heads,
/// head dim, vocab, positions T, rope theta, RMSNorm epsilon, BF16 norm flag,
/// n_out = predictions R. The prompt is the first P = T - R + 1 positions.
/// Real layer records have 1 <= R <= T; ModelID uses the template (0, 0).
pub fn layer_profile(positions: u32, predictions: u32) -> [u8; 64] {
    let fields = [
        1,
        DIM,
        9728,
        32,
        8,
        128,
        VOCAB,
        positions,
        1_000_000f32.to_bits(),
        1e-6f32.to_bits(),
        32, // qwen_common::BF16_SERVING
        predictions,
    ];
    let mut out = [0; 64];
    for (o, f) in out.chunks_exact_mut(4).zip(fields) {
        o.copy_from_slice(&f.to_le_bytes());
    }
    out
}

/// Checkpoint tensor names consumed by a stage, in enrollment order.
pub fn stage_tensors(role: Role) -> Vec<String> {
    let tied = "model.embed_tokens.weight".to_string();
    match role {
        Role::Embed => vec![tied],
        Role::Layer(i) => LAYER_TENSORS
            .iter()
            .map(|t| format!("model.layers.{i}.{t}.weight"))
            .collect(),
        Role::Head => vec!["model.norm.weight".to_string(), tied],
    }
}

/// Root over the SHA-256 of each consumed tensor's raw checkpoint bytes, in
/// [`stage_tensors`] order. These are the enrollment inventory's digests.
pub fn weights_root(role: Role, tensors: &[Hash]) -> Hash {
    let (kind, index) = role.code();
    let mut h = start(b"stage-weights");
    h.update([kind]);
    h.update(index.to_le_bytes());
    h.update((tensors.len() as u32).to_le_bytes());
    for t in tensors {
        h.update(t);
    }
    h.finalize().into()
}

pub fn model_id(checkpoint: &Hash, weights: &[Hash]) -> Hash {
    let mut h = start(b"model/qwen3-4b/mac-arm64-bf16-sdpa-v1");
    h.update(checkpoint);
    h.update(layer_profile(0, 0));
    h.update((weights.len() as u32).to_le_bytes());
    for w in weights {
        h.update(w);
    }
    h.finalize().into()
}

pub struct Request<'a> {
    pub nonce: Hash,
    pub prompt: &'a [u32],
    pub max_reply: u32,
}

#[derive(Debug, PartialEq, Eq)]
pub enum End {
    Stop,
    Length,
}

pub struct Statement {
    pub request_id: Hash,
    pub reply_root: Hash,
    pub end: End,
}

#[derive(Debug, PartialEq, Eq)]
pub enum Error {
    Policy,
    Length,
    Token,
    Termination,
    Roster,
    Proof,
    Receipt,
    Weights,
}

fn start(domain: &[u8]) -> Sha256 {
    let mut h = Sha256::new();
    h.update(b"PCI/v1/");
    h.update(domain);
    h.update([0]);
    h
}

fn ids(domain: &[u8], first: &[u32], second: &[u32]) -> Hash {
    let mut h = start(domain);
    h.update((first.len() as u64 + second.len() as u64).to_le_bytes());
    for id in first.iter().chain(second) {
        h.update(id.to_le_bytes());
    }
    h.finalize().into()
}

pub fn request_id(release: &Release, request: &Request) -> Hash {
    let mut h = start(b"request/greedy-lowest-id");
    h.update(release.model());
    h.update(release.suite);
    h.update(request.nonce);
    h.update(ids(b"prompt", request.prompt, &[]));
    h.update(request.max_reply.to_le_bytes());
    h.update(release.context.to_le_bytes());
    h.update(ids(b"stop-ids", &release.stop_ids, &[]));
    h.finalize().into()
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Receipt {
    pub role: Role,
    pub positions: u32,
    /// For layers and the head these specify the prediction window. Zero for the embedding.
    pub first_prediction: u32,
    pub predictions: u32,
    /// Embedding consumes a token root; layers and head consume activation roots.
    pub input: Hash,
    /// Head produces a reply root; embedding and layers produce activation roots.
    pub output: Hash,
    /// [`weights_root`] of the tensors the guest hashed and consumed.
    pub weights: Hash,
}

impl Role {
    fn code(self) -> (u8, u32) {
        match self {
            Role::Embed => (0, 0),
            Role::Layer(i) => (1, i),
            Role::Head => (2, 0),
        }
    }
}

impl Receipt {
    pub fn digest(&self, model: Hash, suite: Hash, request: Hash) -> Hash {
        let mut h = start(b"child");
        h.update(model);
        h.update(suite);
        h.update(request);
        let (kind, index) = self.role.code();
        h.update([kind]);
        for v in [
            index,
            self.positions,
            self.first_prediction,
            self.predictions,
        ] {
            h.update(v.to_le_bytes());
        }
        h.update(self.input);
        h.update(self.output);
        h.update(self.weights);
        h.finalize().into()
    }
}

/// Commit a row-major Qwen3-4B residual stream, preserving every FP32 bit.
pub fn activation_root(
    model: Hash,
    request: Hash,
    boundary: u32,
    positions: u32,
    values: &[f32],
) -> Result<Hash, Error> {
    if u64::from(positions) * u64::from(DIM) != values.len() as u64 {
        return Err(Error::Length);
    }
    let mut h = Sha256::new();
    h.update(activation_prefix(model, request, boundary, positions)?);
    for v in values {
        h.update(v.to_bits().to_le_bytes());
    }
    Ok(h.finalize().into())
}

/// Bytes preceding the raw little-endian FP32 values in an activation root's
/// SHA-256 preimage. Guests hash `prefix || values` with the accelerated hash.
pub fn activation_prefix(
    model: Hash,
    request: Hash,
    boundary: u32,
    positions: u32,
) -> Result<Vec<u8>, Error> {
    if boundary > LAYERS as u32 || !(1..=40_960).contains(&positions) {
        return Err(Error::Length);
    }
    let mut out = b"PCI/v1/activation\0".to_vec();
    out.extend(model);
    out.extend(request);
    for v in [boundary, positions, DIM] {
        out.extend(v.to_le_bytes());
    }
    Ok(out)
}

pub fn consumed_root(prompt: &[u32], feedback: &[u32]) -> Hash {
    ids(b"consumed", prompt, feedback)
}
pub fn reply_root(reply: &[u32]) -> Hash {
    ids(b"reply", reply, &[])
}

/// Implement this only with a trusted backend adapter. Verify the proof kind,
/// setup key, and expected role's program VK from the pinned suite. Return the
/// authenticated guest bytes after enforcing a 32-byte digest and zero padding.
pub trait ProofBackend {
    type Proof;
    fn verify_child(
        &self,
        suite: &Hash,
        expected: Role,
        proof: &Self::Proof,
    ) -> Result<Hash, Error>;
}

pub fn verify_bundle<B: ProofBackend>(
    backend: &B,
    release: &Release,
    req: &Request,
    reply: &[u32],
    children: &[(Receipt, B::Proof)],
) -> Result<Statement, Error> {
    if !(1..=40_960).contains(&release.context)
        || release.weights.len() != STAGES
        || release.stop_ids.iter().any(|id| *id >= VOCAB)
        || !release.stop_ids.windows(2).all(|w| w[0] < w[1])
    {
        return Err(Error::Policy);
    }
    let p = u32::try_from(req.prompt.len()).map_err(|_| Error::Length)?;
    let r = u32::try_from(reply.len()).map_err(|_| Error::Length)?;
    if p == 0
        || r == 0
        || r > req.max_reply
        || p.checked_add(req.max_reply)
            .is_none_or(|v| v > release.context)
    {
        return Err(Error::Length);
    }
    if req.prompt.iter().chain(reply).any(|id| *id >= VOCAB) {
        return Err(Error::Token);
    }
    let (&last, feedback) = reply.split_last().ok_or(Error::Length)?;
    let stop = |id: &u32| release.stop_ids.binary_search(id).is_ok();
    if feedback.iter().any(stop) {
        return Err(Error::Termination);
    }
    let end = if stop(&last) {
        End::Stop
    } else if r == req.max_reply {
        End::Length
    } else {
        return Err(Error::Termination);
    };
    if children.len() != STAGES {
        return Err(Error::Roster);
    }
    let positions = p + r - 1;
    let model = release.model();
    let request_id = request_id(release, req);
    let reply_root = reply_root(reply);
    let mut previous = consumed_root(req.prompt, feedback);
    for (i, (receipt, proof)) in children.iter().enumerate() {
        let role = match i {
            0 => Role::Embed,
            1..=LAYERS => Role::Layer((i - 1) as u32),
            _ => Role::Head,
        };
        if receipt.role != role {
            return Err(Error::Roster);
        }
        let digest = backend
            .verify_child(&release.suite, role, proof)
            .map_err(|_| Error::Proof)?;
        let window = if role == Role::Embed { (0, 0) } else { (p - 1, r) };
        if receipt.weights != release.weights[i] {
            return Err(Error::Weights);
        }
        if digest != receipt.digest(model, release.suite, request_id)
            || receipt.positions != positions
            || (receipt.first_prediction, receipt.predictions) != window
            || receipt.input != previous
        {
            return Err(Error::Receipt);
        }
        previous = receipt.output;
    }
    if previous != reply_root {
        return Err(Error::Receipt);
    }
    Ok(Statement {
        request_id,
        reply_root,
        end,
    })
}
