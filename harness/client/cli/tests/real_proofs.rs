//! Tests on real artifacts: the Qwen3-4B checkpoint and the GPU proofs written
//! by `~/scale-runs/pci-client/prove-1x1.sh`. Ignored by default because they
//! need those files; run with `cargo test --release -p pci-client -- --ignored`.
use pci_client::*;
use pci_receipts::DIM;
use pci_stages::{head, rows};
use serde_json::Value;
use std::path::{Path, PathBuf};

const RUN: &str = "/home/abix/scale-runs/pci-client";
const MODEL: &str = "/home/abix/qwen-profile-20261119/qwen3-4b";
const INVENTORY: &str = "/home/abix/scale-runs/pci-layer-binding/inventory.json";

fn art(p: &str) -> PathBuf {
    Path::new(RUN).join(p)
}

fn json(p: &Path) -> Value {
    serde_json::from_slice(&std::fs::read(p).unwrap()).unwrap()
}

fn expected_digest(name: &str) -> Hash {
    let e = json(&art("fixture-1x1/expected.json"));
    let c = e["children"].as_array().unwrap().iter().find(|c| c["name"] == name).unwrap();
    unhex(c["digest"].as_str().unwrap()).unwrap()
}

type Hash = [u8; 32];

/// Every logit of every prediction from the 297 shard executions must equal
/// pci-layer-host's native full head on the same A36, and the combined greedy
/// IDs must be the transcript's reply (token feedback included for R > 1).
fn sharded_head_matches(a36_file: &str, logits_file: &str, r: u32, reply: &[u32], dotf8: bool) {
    let inv = Inventory::load(Path::new(INVENTORY)).unwrap();
    let ck = Checkpoint::open(Path::new(MODEL)).unwrap();
    let tied = ck.authenticated(&inv, TIED).unwrap();
    let norm = ck.authenticated(&inv, NORM).unwrap();
    let tree = rows::Tree::build(sha, &tied);
    let pinned = json(&art("rows.json"));
    assert_eq!(hex(&tree.root(sha)), pinned["rows_root"].as_str().unwrap());
    let a36: Vec<f32> = std::fs::read(a36_file)
        .unwrap()
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect();
    let want = std::fs::read(logits_file).unwrap();
    let vocab = pci_receipts::VOCAB as usize;
    assert_eq!(want.len(), 4 * vocab * r as usize);
    let mut got = vec![0u8; want.len()];
    let mut receipts = vec![];
    for k in 0..head::shard_count(9) {
        let rec = head::encode([1; 32], [2; 32], [3; 32], 9, k, r, &a36, &norm, &tied, &tree);
        let o = if dotf8 {
            let mut kern = pci_dotf8::Kernel::new(pci_dotf8::SoftDotF8);
            let mut f = |x: &[u32], m: &[u8], out: &mut [f32]| kern.project(&pci_dotf8::Prepared::new(x, DIM as usize), m, out);
            head::execute_with(&rec, sha, &mut f).unwrap()
        } else {
            head::execute(&rec, sha).unwrap()
        };
        let n = head::shard_rows(9, k) as usize;
        for (j, l) in o.logits.chunks_exact(n).enumerate() {
            let at = 4 * (j * vocab + ((k as usize) << 9));
            got[at..at + 4 * n].copy_from_slice(&l.iter().flat_map(|v| v.to_bits().to_le_bytes()).collect::<Vec<_>>());
        }
        receipts.push(o.receipt);
    }
    assert_eq!(a36.len(), (receipts[0].positions * DIM) as usize);
    assert!(got == want, "logits differ from pci-layer-host");
    assert_eq!(head::combine(&receipts).unwrap(), reply);
}

#[test]
#[ignore]
fn real_checkpoint_sharded_head_matches_layer_host_logits() {
    // P=1, R=1 (A36 from the binding run, whose layers were also emulated).
    sharded_head_matches(
        "/home/abix/scale-runs/pci-layer-binding/full-1x1/a36.f32",
        "/home/abix/scale-runs/pci-layer-binding/full-1x1/logits.f32",
        1,
        &[279],
        false,
    );
}

#[test]
#[ignore]
fn real_checkpoint_three_token_transcript() {
    // P=1, R=3 greedy transcript 12522 -> 279, 279, 1196 (native layers).
    sharded_head_matches(
        &format!("{RUN}/native-1x3/final/a36.f32"),
        &format!("{RUN}/native-1x3/final/logits.f32"),
        3,
        &[279, 279, 1196],
        false,
    );
}

#[test]
#[ignore]
fn real_checkpoint_three_token_transcript_batched_dotf8_kernel() {
    // Same, with batch's batched lossless dot_f8 kernel (software dot_f8 backend).
    sharded_head_matches(
        &format!("{RUN}/native-1x3/final/a36.f32"),
        &format!("{RUN}/native-1x3/final/logits.f32"),
        3,
        &[279, 279, 1196],
        true,
    );
}

#[test]
#[ignore]
fn real_proofs_reload_and_verify_under_pins() {
    let pins = Pins::from_json(&json(&art("release.json"))).unwrap();
    let zisk = Zisk { pins: &pins };
    for (name, kind, approved) in [
        ("embed", Kind::Embed, &pins.embed_programs),
        ("head-00000", Kind::Head, &pins.head_programs),
    ] {
        let proof = std::fs::read(art(&format!("proofs/{name}.proof.bin"))).unwrap();
        // Reloaded from disk, verified under the pinned setup key and program.
        let facts = verify_zisk_leaf(&proof, &pins.setup_vk, approved).unwrap();
        assert_eq!(facts.digest, expected_digest(name), "{name}");
        assert_eq!(zisk.verify(kind, &proof).unwrap(), expected_digest(name));

        // The other role's program list does not approve it.
        let other = if kind == Kind::Embed { Kind::Head } else { Kind::Embed };
        assert!(zisk.verify(other, &proof).unwrap_err().contains("not approved"), "{name}");
        // A different setup key.
        let mut k = pins.setup_vk;
        k[0] ^= 1;
        assert!(verify_zisk_leaf(&proof, &k, approved).is_err(), "{name} wrong setup key");
        // A different approved program VK with the right key.
        let mut a = approved[0];
        a[3] ^= 1;
        assert!(verify_zisk_leaf(&proof, &pins.setup_vk, &[a]).is_err(), "{name} wrong program");

        // Flat layout: [minimal][n_publics][flag][program vk 4][outputs 64][stark...][vk 4][tag].
        let word = |w: usize| 8 * w;
        let mutate = |at: usize, x: u8| {
            let mut b = proof.clone();
            b[at] ^= x;
            b
        };
        for (what, bad) in [
            ("program vk word", mutate(word(3), 1)),
            ("first digest word", mutate(word(7), 1)),
            ("a zero output word", mutate(word(7 + 20), 1)),
            ("stark body", mutate(proof.len() / 2, 1)),
            ("claimed vk (unauthenticated) and tag", mutate(proof.len() - 9, 1)),
            ("truncated", proof[..proof.len() - 8].to_vec()),
            ("trailing word", [proof.clone(), vec![0; 8]].concat()),
        ] {
            let r = verify_zisk_leaf(&bad, &pins.setup_vk, approved);
            // The claimed-vk word is metadata the verifier ignores; anything else must reject.
            if what.starts_with("claimed vk") {
                continue;
            }
            assert!(r.is_err(), "{name}: {what} accepted");
        }
    }
}

#[test]
#[ignore]
fn real_partial_bundle_is_not_accepted() {
    // Embedding and one head shard are real proofs; layers and 296 shards are
    // absent, so the verifier must reject at the roster before anything else.
    let pins = Pins::from_json(&json(&art("release.json"))).unwrap();
    let req = ClientRequest::from_json(&json(&art("request-1x1.json"))).unwrap();
    let e = json(&art("fixture-1x1/expected.json"));
    let dir = std::env::temp_dir().join(format!("pci-partial-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let mut children = vec![];
    for name in ["embed", "head-00000"] {
        let c = e["children"].as_array().unwrap().iter().find(|c| c["name"] == name).unwrap();
        let f = format!("{name}.proof.bin");
        std::fs::copy(art(&format!("proofs/{f}")), dir.join(&f)).unwrap();
        children.push((c["receipt"].clone(), f));
    }
    std::fs::write(dir.join("bundle.json"), serde_json::to_vec(&Bundle::to_json(&[279], &children)).unwrap()).unwrap();
    let b = Bundle::load(&dir).unwrap();
    let r = verify_bundle(&Zisk { pins: &pins }, &pins, &req, &b);
    assert!(matches!(r, Err(Reject::Roster(_))), "{r:?}");
    std::fs::remove_dir_all(&dir).unwrap();
}
