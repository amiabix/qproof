//! Embedding and vocabulary-sharded head children for the token-ID,
//! ordered-FP32, greedy Qwen3-4B suite. The guests and the native host run the
//! same functions; only the SHA-256 implementation differs.
//!
//! Weight access. Both stages read the tied `model.embed_tokens.weight` matrix
//! (151,936 rows of 2,560 BF16 = 5,120 bytes). Its whole-tensor SHA-256 is the
//! enrollment inventory entry; guests cannot afford to hash 778 MB per child,
//! so they open rows under a [`rows`] Merkle commitment instead. The client
//! derives that root itself from checkpoint bytes whose SHA-256 it has checked
//! against the pinned inventory (`pci-client rows`), and pins it.
//!
//! Every hash is SHA-256 over `"PCI/v1/" || domain || 0 || fields`, with
//! fixed-width little-endian integers; the domains here do not occur in
//! `pci_receipts`.
pub use pci_receipts::{DIM, Hash, LAYERS, VOCAB};

pub type Sha = fn(&[u8]) -> Hash;

/// Bytes of one BF16 row of the tied matrix.
pub const ROW_BYTES: usize = 2 * DIM as usize;

pub fn tagged(sha: Sha, domain: &[u8], parts: &[&[u8]]) -> Hash {
    let mut pre = Vec::with_capacity(8 + domain.len() + parts.iter().map(|p| p.len()).sum::<usize>());
    pre.extend_from_slice(b"PCI/v1/");
    pre.extend_from_slice(domain);
    pre.push(0);
    for p in parts {
        pre.extend_from_slice(p);
    }
    sha(&pre)
}

pub(crate) fn u32_at(b: &[u8], at: usize) -> u32 {
    u32::from_le_bytes(b[at..at + 4].try_into().unwrap())
}

pub(crate) fn hash_at(b: &[u8], at: usize) -> Hash {
    b[at..at + 32].try_into().unwrap()
}

/// Exact BF16 to FP32 bit widening; rejects nonfinite values.
pub(crate) fn widen_finite(bf16: &[u8], out: &mut Vec<u8>) -> Result<(), &'static str> {
    for p in bf16.chunks_exact(2) {
        let v = u16::from_le_bytes([p[0], p[1]]);
        if (v >> 7) & 0xff == 0xff {
            return Err("nonfinite weight");
        }
        out.extend_from_slice(&(u32::from(v) << 16).to_le_bytes());
    }
    Ok(())
}

pub mod rows {
    //! Fixed-depth binary Merkle tree over the tied matrix rows.
    //!
    //! ```text
    //! leaf(i)  = H("row",      i u32 || 5120 raw BF16 bytes)     i < 151936
    //! leaf(i)  = 32 zero bytes                                   151936 <= i < 2^18
    //! node     = H("row-node", left || right)
    //! root     = H("rows-root", VOCAB u32 || ROW_BYTES u32 || DEPTH u32 || top)
    //! ```
    use super::*;

    pub const DEPTH: u32 = 18;
    const _: () = assert!(VOCAB <= 1 << DEPTH);

    pub fn leaf(sha: Sha, index: u32, row: &[u8]) -> Hash {
        debug_assert_eq!(row.len(), ROW_BYTES);
        tagged(sha, b"row", &[&index.to_le_bytes(), row])
    }

    pub fn node(sha: Sha, left: &Hash, right: &Hash) -> Hash {
        tagged(sha, b"row-node", &[left, right])
    }

    pub fn root(sha: Sha, top: &Hash) -> Hash {
        tagged(
            sha,
            b"rows-root",
            &[
                &VOCAB.to_le_bytes(),
                &(ROW_BYTES as u32).to_le_bytes(),
                &DEPTH.to_le_bytes(),
                top,
            ],
        )
    }

    /// Climb from node `index` at `level` to the top using `siblings`
    /// (bottom-up, `DEPTH - level` of them) and wrap the result as a root.
    pub fn climb(sha: Sha, mut h: Hash, mut index: u32, level: u32, siblings: &[u8]) -> Hash {
        assert_eq!(siblings.len(), 32 * (DEPTH - level) as usize);
        for s in siblings.chunks_exact(32) {
            let s: &Hash = s.try_into().unwrap();
            h = if index & 1 == 0 {
                node(sha, &h, s)
            } else {
                node(sha, s, &h)
            };
            index >>= 1;
        }
        root(sha, &h)
    }

    /// The whole tree, for the host side. `levels[0]` holds the 151,936 real
    /// leaves; missing right siblings are the empty-subtree hashes.
    pub struct Tree {
        pub levels: Vec<Vec<Hash>>,
        empty: Vec<Hash>,
    }

    impl Tree {
        pub fn build(sha: Sha, tied: &[u8]) -> Self {
            assert_eq!(tied.len(), VOCAB as usize * ROW_BYTES);
            let leaves = tied
                .chunks_exact(ROW_BYTES)
                .enumerate()
                .map(|(i, r)| leaf(sha, i as u32, r))
                .collect();
            Self::from_leaves(sha, leaves)
        }

        pub fn from_leaves(sha: Sha, leaves: Vec<Hash>) -> Self {
            assert_eq!(leaves.len(), VOCAB as usize);
            let mut empty = vec![[0u8; 32]];
            for l in 0..DEPTH as usize {
                empty.push(node(sha, &empty[l], &empty[l]));
            }
            let mut levels = vec![leaves];
            for l in 0..DEPTH as usize {
                let next = levels[l]
                    .chunks(2)
                    .map(|p| node(sha, &p[0], p.get(1).unwrap_or(&empty[l])))
                    .collect();
                levels.push(next);
            }
            assert_eq!(levels[DEPTH as usize].len(), 1);
            Tree { levels, empty }
        }

        pub fn root(&self, sha: Sha) -> Hash {
            root(sha, &self.levels[DEPTH as usize][0])
        }

        fn at(&self, level: usize, index: usize) -> Hash {
            *self.levels[level].get(index).unwrap_or(&self.empty[level])
        }

        /// Siblings of node `index` at `level`, bottom-up, as raw bytes.
        pub fn path(&self, level: u32, mut index: u32) -> Vec<u8> {
            let mut out = Vec::with_capacity(32 * (DEPTH - level) as usize);
            for l in level..DEPTH {
                out.extend_from_slice(&self.at(l as usize, (index ^ 1) as usize));
                index >>= 1;
            }
            out
        }
    }
}

/// Weights root of the embedding child: the row commitment it opened.
pub fn embed_weights(sha: Sha, rows_root: &Hash) -> Hash {
    tagged(sha, b"embed-weights/rows-v1", &[rows_root])
}

/// Weights root of every head shard: SHA-256 of the raw final-norm bytes and
/// the row commitment.
pub fn head_weights(sha: Sha, norm_sha: &Hash, rows_root: &Hash) -> Hash {
    tagged(sha, b"head-weights/rows-v1", &[norm_sha, rows_root])
}

fn activation_root(
    sha: Sha,
    model: Hash,
    request: Hash,
    boundary: u32,
    positions: u32,
    values: &[u8],
) -> Result<Hash, &'static str> {
    let mut pre = pci_receipts::activation_prefix(model, request, boundary, positions)
        .map_err(|_| "activation shape")?;
    pre.extend_from_slice(values);
    Ok(sha(&pre))
}

pub mod embed {
    //! Record v1 (little-endian, no trailing bytes):
    //! ```text
    //! "PCIEMBD1" | model 32 | suite 32 | request 32 | T u32 | T token IDs u32
    //! | per position: row 5120 raw BF16 bytes | 18 siblings x 32
    //! ```
    //! The receipt is a `pci_receipts::Receipt` of role Embed: input is the
    //! consumed-token root of the T IDs, output the boundary-0 activation root
    //! of the exactly widened rows, weights [`embed_weights`] of the single
    //! row root every opening reproduced.
    use super::*;
    use pci_receipts::{Receipt, Role, consumed_root};

    pub const MAGIC: &[u8; 8] = b"PCIEMBD1";
    pub const FIXED: usize = 8 + 3 * 32 + 4;
    const OPENING: usize = ROW_BYTES + 32 * rows::DEPTH as usize;

    pub struct Outcome {
        pub receipt: Receipt,
        pub rows_root: Hash,
        pub digest: Hash,
        /// A0 as little-endian FP32 bytes.
        pub output: Vec<u8>,
    }

    pub fn execute(record: &[u8], sha: Sha) -> Result<Outcome, &'static str> {
        if record.len() < FIXED || &record[..8] != MAGIC {
            return Err("magic");
        }
        let (model, suite, request) = (hash_at(record, 8), hash_at(record, 40), hash_at(record, 72));
        let t = u32_at(record, 104);
        if !(1..=40_960).contains(&t) {
            return Err("positions");
        }
        let n = t as usize;
        if record.len() != FIXED + 4 * n + n * OPENING {
            return Err("record length");
        }
        let ids: Vec<u32> = (0..n).map(|i| u32_at(record, FIXED + 4 * i)).collect();
        if ids.iter().any(|&id| id >= VOCAB) {
            return Err("token id out of range");
        }
        let mut root: Option<Hash> = None;
        let mut a0 = Vec::with_capacity(n * DIM as usize * 4);
        let base = FIXED + 4 * n;
        for (i, &id) in ids.iter().enumerate() {
            let o = &record[base + i * OPENING..][..OPENING];
            let (row, siblings) = o.split_at(ROW_BYTES);
            let r = rows::climb(sha, rows::leaf(sha, id, row), id, 0, siblings);
            if *root.get_or_insert(r) != r {
                return Err("openings disagree on the row root");
            }
            widen_finite(row, &mut a0)?;
        }
        let rows_root = root.unwrap();
        let receipt = Receipt {
            role: Role::Embed,
            positions: t,
            first_prediction: 0,
            predictions: 0,
            input: consumed_root(&ids, &[]),
            output: activation_root(sha, model, request, 0, t, &a0)?,
            weights: embed_weights(sha, &rows_root),
        };
        let digest = receipt.digest(model, suite, request);
        Ok(Outcome {
            receipt,
            rows_root,
            digest,
            output: a0,
        })
    }

    pub fn encode(
        model: Hash,
        suite: Hash,
        request: Hash,
        ids: &[u32],
        tied: &[u8],
        tree: &rows::Tree,
    ) -> Vec<u8> {
        let mut r = MAGIC.to_vec();
        for h in [model, suite, request] {
            r.extend(h);
        }
        r.extend((ids.len() as u32).to_le_bytes());
        for id in ids {
            r.extend(id.to_le_bytes());
        }
        for &id in ids {
            r.extend_from_slice(&tied[id as usize * ROW_BYTES..][..ROW_BYTES]);
            r.extend(tree.path(0, id));
        }
        r
    }
}

pub mod head {
    //! One aligned vocabulary shard of the full head.
    //!
    //! Record v1 (little-endian, no trailing bytes):
    //! ```text
    //! "PCIHEAD1" | model 32 | suite 32 | request 32 | shard_log2 u32 | shard u32
    //! | T u32 | R u32 | A36: T x 2560 FP32 bits | final norm 5120 raw BF16 bytes
    //! | rows [shard*2^s, min((shard+1)*2^s, VOCAB)) raw BF16 | (18 - s) siblings x 32
    //! ```
    //! For each of the last R positions the guest applies the final RMSNorm
    //! (`qwen_common::run_with`, HEAD segment, ordered FP32) and computes every
    //! logit of the shard with the reference projection (`pci_layer::project`:
    //! sequential FP32 from +0 over exactly widened BF16). Any nonfinite logit
    //! rejects. The shard winner is the lowest ID among the largest values
    //! (`>` comparison, so +0 and -0 tie). The receipt binds the whole window.
    use super::*;
    use qwen_common::{HEAD, HEADER_BYTES, Header, run_with};
    use std::cell::RefCell;

    pub const MAGIC: &[u8; 8] = b"PCIHEAD1";
    pub const FIXED: usize = 8 + 3 * 32 + 4 * 4;
    /// Shard sizes the suite admits.
    pub const MIN_LOG2: u32 = 4;
    pub const MAX_LOG2: u32 = rows::DEPTH;

    pub fn shard_count(log2: u32) -> u32 {
        VOCAB.div_ceil(1 << log2)
    }

    pub fn shard_rows(log2: u32, shard: u32) -> u32 {
        (1u32 << log2).min(VOCAB - (shard << log2))
    }

    #[derive(Clone, Debug, PartialEq, Eq)]
    pub struct ShardReceipt {
        pub shard_log2: u32,
        pub shard: u32,
        pub positions: u32,
        pub first_prediction: u32,
        pub predictions: u32,
        /// Boundary-36 activation root.
        pub input: Hash,
        /// [`head_weights`].
        pub weights: Hash,
        /// Per prediction: (winning token ID in this shard, its logit bits).
        pub best: Vec<(u32, u32)>,
    }

    impl ShardReceipt {
        pub fn digest(&self, sha: Sha, model: Hash, suite: Hash, request: Hash) -> Hash {
            let mut fields = Vec::with_capacity(5 * 4 + 64 + 8 * self.best.len());
            for v in [
                self.shard_log2,
                self.shard,
                self.positions,
                self.first_prediction,
                self.predictions,
            ] {
                fields.extend(v.to_le_bytes());
            }
            fields.extend(self.input);
            fields.extend(self.weights);
            fields.extend((self.best.len() as u32).to_le_bytes());
            for (id, bits) in &self.best {
                fields.extend(id.to_le_bytes());
                fields.extend(bits.to_le_bytes());
            }
            tagged(sha, b"head-shard/v1", &[&model, &suite, &request, &fields])
        }
    }

    pub struct Outcome {
        pub receipt: ShardReceipt,
        pub rows_root: Hash,
        pub digest: Hash,
        /// R x shard rows logits, for diagnostics.
        pub logits: Vec<f32>,
    }

    /// The 64-byte HEAD segment header for a shard of `rows` rows.
    pub fn header(positions: u32, predictions: u32, rows: u32) -> Header {
        Header {
            kind: HEAD,
            dim: DIM as usize,
            hidden: 9728,
            heads: 32,
            kv_heads: 8,
            head_dim: 128,
            vocab: rows as usize,
            n_tok: positions as usize,
            rope_theta: 1e6,
            eps: 1e-6,
            flags: 0,
            n_out: predictions as usize,
        }
    }

    /// A batched projection kernel: `x` holds `positions x 2560` FP32 bit
    /// patterns, `out` receives `positions x rows` logits (`out[t * rows + r]`).
    /// Every kernel must be bit-identical to [`reference_kernel`].
    pub type Kernel<'a> = &'a mut dyn FnMut(&[u32], &[u8], &mut [f32]);

    /// `pci_layer::project` (sequential FP32 from +0 over exactly widened BF16)
    /// for each position.
    pub fn reference_kernel(x: &[u32], matrix: &[u8], out: &mut [f32]) {
        let cols = DIM as usize;
        let rows = matrix.len() / (2 * cols);
        for (xt, ot) in x.chunks_exact(cols).zip(out.chunks_exact_mut(rows)) {
            pci_layer::project_bits(xt, matrix, ot);
        }
    }

    pub fn execute(record: &[u8], sha: Sha) -> Result<Outcome, &'static str> {
        execute_with(record, sha, &mut reference_kernel)
    }

    /// As [`execute`] with a caller-supplied batched projection kernel.
    pub fn execute_with(record: &[u8], sha: Sha, kernel: Kernel) -> Result<Outcome, &'static str> {
        if record.len() < FIXED || &record[..8] != MAGIC {
            return Err("magic");
        }
        let (model, suite, request) = (hash_at(record, 8), hash_at(record, 40), hash_at(record, 72));
        let (log2, shard, t, r) = (
            u32_at(record, 104),
            u32_at(record, 108),
            u32_at(record, 112),
            u32_at(record, 116),
        );
        if !(MIN_LOG2..=MAX_LOG2).contains(&log2) || shard >= shard_count(log2) {
            return Err("shard");
        }
        if !(1..=40_960).contains(&t) || r == 0 || r > t {
            return Err("positions");
        }
        let rows = shard_rows(log2, shard) as usize;
        let act = t as usize * DIM as usize * 4;
        let sib = 32 * (rows::DEPTH - log2) as usize;
        if record.len() != FIXED + act + ROW_BYTES + rows * ROW_BYTES + sib {
            return Err("record length");
        }
        let x = &record[FIXED..FIXED + act];
        let norm = &record[FIXED + act..][..ROW_BYTES];
        let matrix = &record[FIXED + act + ROW_BYTES..][..rows * ROW_BYTES];
        let siblings = &record[FIXED + act + ROW_BYTES + rows * ROW_BYTES..];

        // Authenticate the shard: hash its rows, pad to 2^s leaves, climb.
        let first_id = shard << log2;
        let mut level: Vec<Hash> = matrix
            .chunks_exact(ROW_BYTES)
            .enumerate()
            .map(|(i, row)| rows::leaf(sha, first_id + i as u32, row))
            .collect();
        // Leaves past the vocabulary are the zero padding leaf.
        level.resize(1 << log2, [0u8; 32]);
        for _ in 0..log2 {
            level = level
                .chunks_exact(2)
                .map(|p| rows::node(sha, &p[0], &p[1]))
                .collect();
        }
        let rows_root = rows::climb(sha, level[0], shard, log2, siblings);
        let weights = head_weights(sha, &sha(norm), &rows_root);
        let input = activation_root(sha, model, request, LAYERS as u32, t, x)?;

        // HEAD segment: header, A36 FP32 bits, final norm widened.
        let h = header(t, r, rows as u32);
        let mut seg = Vec::with_capacity(HEADER_BYTES + act + 4 * DIM as usize);
        seg.extend_from_slice(&h.to_bytes());
        for v in x.chunks_exact(4) {
            if (u32_at(v, 0) >> 23) & 0xff == 0xff {
                return Err("nonfinite input activation");
            }
        }
        seg.extend_from_slice(x);
        widen_finite(norm, &mut seg)?;
        // run_with reads u32 words; keep the segment 4-byte aligned.
        let words: Vec<u32> = seg.chunks_exact(4).map(|b| u32_at(b, 0)).collect();
        // SAFETY: u32 to bytes reinterpretation; `words` outlives the slice.
        let seg = unsafe { std::slice::from_raw_parts(words.as_ptr().cast::<u8>(), 4 * words.len()) };

        // Final RMSNorm of the last R positions through qwen_common's HEAD
        // segment; the projection is then one batched kernel call so each
        // vocabulary row is decoded once for all R positions.
        let xn = RefCell::new(Vec::<u32>::with_capacity(r as usize * DIM as usize));
        let mut observe = |_: usize, v: &[f32]| xn.borrow_mut().extend(v.iter().map(|f| f.to_bits()));
        let mut claim = |_: usize, out: &mut [f32]| out.fill(0.0);
        let qdot_unused: qwen_common::QDotFn = |_, _, _, _, _| panic!("quantized path disabled");
        run_with(seg, qwen_common::dot_soft, qdot_unused, &mut observe, Some(&mut claim));
        let xn = xn.into_inner();
        if xn.len() != r as usize * DIM as usize {
            return Err("normalized input shape");
        }
        let mut logits = vec![0.0f32; r as usize * rows];
        kernel(&xn, matrix, &mut logits);
        if logits.len() != r as usize * rows || logits.iter().any(|v| !v.is_finite()) {
            return Err("nonfinite or misshapen logits");
        }
        let best = logits
            .chunks_exact(rows)
            .map(|l| {
                let mut b = 0;
                for (i, v) in l.iter().enumerate() {
                    if *v > l[b] {
                        b = i;
                    }
                }
                (first_id + b as u32, l[b].to_bits())
            })
            .collect();
        let receipt = ShardReceipt {
            shard_log2: log2,
            shard,
            positions: t,
            first_prediction: t - r,
            predictions: r,
            input,
            weights,
            best,
        };
        let digest = receipt.digest(sha, model, suite, request);
        Ok(Outcome {
            receipt,
            rows_root,
            digest,
            logits,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn encode(
        model: Hash,
        suite: Hash,
        request: Hash,
        log2: u32,
        shard: u32,
        predictions: u32,
        a36: &[f32],
        norm: &[u8],
        tied: &[u8],
        tree: &rows::Tree,
    ) -> Vec<u8> {
        let t = (a36.len() / DIM as usize) as u32;
        let mut r = MAGIC.to_vec();
        for h in [model, suite, request] {
            r.extend(h);
        }
        for v in [log2, shard, t, predictions] {
            r.extend(v.to_le_bytes());
        }
        r.extend(a36.iter().flat_map(|v| v.to_bits().to_le_bytes()));
        r.extend_from_slice(norm);
        let first = (shard << log2) as usize;
        let rows = shard_rows(log2, shard) as usize;
        r.extend_from_slice(&tied[first * ROW_BYTES..(first + rows) * ROW_BYTES]);
        r.extend(tree.path(log2, shard));
        r
    }

    /// Greedy IDs over the full vocabulary from every shard's winners, lowest
    /// ID on ties. `shards` must be all shards in order; the caller checks it.
    pub fn combine(shards: &[ShardReceipt]) -> Result<Vec<u32>, &'static str> {
        let first = shards.first().ok_or("no shards")?;
        let mut best: Vec<(u32, f32)> = first
            .best
            .iter()
            .map(|&(id, bits)| (id, f32::from_bits(bits)))
            .collect();
        for s in &shards[1..] {
            if s.best.len() != best.len() {
                return Err("prediction count");
            }
            for (b, &(id, bits)) in best.iter_mut().zip(&s.best) {
                let v = f32::from_bits(bits);
                if !v.is_finite() {
                    return Err("nonfinite logit");
                }
                if v > b.1 {
                    *b = (id, v);
                }
            }
        }
        if best.iter().any(|b| !b.1.is_finite()) {
            return Err("nonfinite logit");
        }
        Ok(best.into_iter().map(|b| b.0).collect())
    }
}
