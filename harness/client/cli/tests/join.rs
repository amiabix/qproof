//! Join-logic tests for `verify_bundle`. The `Echo` backend below treats a
//! proof's bytes as its digest. It exists only to drive the roster, link,
//! window, termination and greedy checks; it is NOT acceptance evidence and
//! no binary uses it. Real-proof tests are in `real_proofs.rs`.
use pci_client::*;
use pci_receipts::{Hash, LAYERS, Receipt, Role, consumed_root, request_id};
use pci_stages::head::{self, ShardReceipt};

struct Echo;
impl ChildVerifier for Echo {
    fn verify(&self, kind: Kind, proof: &[u8]) -> Res<Hash> {
        // The first byte names the kind so a proof of one role cannot pass as another.
        let tag = match kind {
            Kind::Embed => 0,
            Kind::Layer => 1,
            Kind::Head => 2,
        };
        if proof.len() != 33 || proof[0] != tag {
            return Err("echo: invalid".into());
        }
        Ok(proof[1..].try_into().unwrap())
    }
}

fn echo(kind: u8, d: Hash) -> Vec<u8> {
    let mut v = vec![kind];
    v.extend(d);
    v
}

fn pins(log2: u32) -> Pins {
    Pins {
        checkpoint: [7; 32],
        stage_weights: (0..38u8).map(|i| [i + 100; 32]).collect(),
        rows_root: [5; 32],
        norm_sha256: [6; 32],
        shard_log2: log2,
        context: 40960,
        stop_ids: vec![151643, 151645],
        setup_vk: [1, 2, 3, 4],
        embed_programs: vec![],
        layer_programs: vec![],
        head_programs: vec![],
    }
}

fn req(prompt: &[u32], max_reply: u32) -> ClientRequest {
    ClientRequest {
        nonce: [9; 32],
        prompt: prompt.to_vec(),
        max_reply,
    }
}

fn root(i: u32) -> Hash {
    let mut h = [0xa0; 32];
    h[..4].copy_from_slice(&i.to_le_bytes());
    h
}

/// An honest bundle under the Echo backend. `logits[k][j]` is shard k's
/// winner (id, value) for prediction j.
fn bundle(p: &Pins, q: &ClientRequest, reply: &[u32], winners: &[Vec<(u32, f32)>]) -> Bundle {
    let rel = p.release();
    let (model, suite) = (rel.model(), rel.suite);
    let request = request_id(&rel, &q.as_request());
    let t = (q.prompt.len() + reply.len() - 1) as u32;
    let r = reply.len() as u32;
    let mut children = vec![];
    let embed = Receipt {
        role: Role::Embed,
        positions: t,
        first_prediction: 0,
        predictions: 0,
        input: consumed_root(&q.prompt, &reply[..reply.len() - 1]),
        output: root(0),
        weights: p.embed_weights(),
    };
    children.push(Child {
        proof: echo(0, embed.digest(model, suite, request)),
        receipt: ChildReceipt::Stage(embed),
        proof_file: "embed".into(),
    });
    for i in 0..LAYERS as u32 {
        let rc = Receipt {
            role: Role::Layer(i),
            positions: t,
            first_prediction: t - r,
            predictions: r,
            input: root(i),
            output: root(i + 1),
            weights: p.stage_weights[i as usize + 1],
        };
        children.push(Child {
            proof: echo(1, rc.digest(model, suite, request)),
            receipt: ChildReceipt::Stage(rc),
            proof_file: format!("layer{i}"),
        });
    }
    for (k, w) in winners.iter().enumerate() {
        let s = ShardReceipt {
            shard_log2: p.shard_log2,
            shard: k as u32,
            positions: t,
            first_prediction: t - r,
            predictions: r,
            input: root(LAYERS as u32),
            weights: p.head_weights(),
            best: w.iter().map(|(id, v)| (*id, v.to_bits())).collect(),
        };
        children.push(Child {
            proof: echo(2, s.digest(sha, model, suite, request)),
            receipt: ChildReceipt::Shard(s),
            proof_file: format!("head{k}"),
        });
    }
    Bundle {
        reply: reply.to_vec(),
        children,
    }
}

/// Re-sign one child after mutating its receipt (as a prover who controls the
/// receipt and can obtain a proof of it would), to reach the semantic checks.
fn resign(p: &Pins, q: &ClientRequest, c: &mut Child) {
    let rel = p.release();
    let request = request_id(&rel, &q.as_request());
    c.proof = match &c.receipt {
        ChildReceipt::Stage(r) => echo(
            if r.role == Role::Embed { 0 } else { 1 },
            r.digest(rel.model(), rel.suite, request),
        ),
        ChildReceipt::Shard(s) => echo(2, s.digest(sha, rel.model(), rel.suite, request)),
    };
}

// s = 17 gives two shards: [0, 131072) and [131072, 151936).
const S: u32 = 17;

fn two_shard_case() -> (Pins, ClientRequest, Bundle) {
    let p = pins(S);
    let q = req(&[12522, 11], 3);
    let reply = [42, 131500, 151645];
    let b = bundle(&p, &q, &reply, &[
        vec![(42, 5.0), (7, 1.0), (9, 0.5)],
        vec![(131072, 4.0), (131500, 2.0), (151645, 3.0)],
    ]);
    (p, q, b)
}

#[test]
fn honest_bundle_accepts() {
    let (p, q, b) = two_shard_case();
    let s = verify_bundle(&Echo, &p, &q, &b).unwrap();
    assert!(s.stopped);
    assert_eq!(s.predicted, vec![42, 131500, 151645]);
    // Length-limited end: no stop token, reply length == max_reply.
    let q = req(&[1], 2);
    let b = bundle(&p, &q, &[3, 4], &[vec![(3, 1.0), (4, 1.0)], vec![(131072, 0.0), (131073, 0.0)]]);
    assert!(!verify_bundle(&Echo, &p, &q, &b).unwrap().stopped);
}

#[test]
fn ties_resolve_to_lowest_id_across_shards() {
    let p = pins(S);
    let q = req(&[1], 1);
    // Equal logits in both shards: the lower ID (shard 0) is the greedy token.
    let b = bundle(&p, &q, &[100], &[vec![(100, 2.0)], vec![(140000, 2.0)]]);
    let b2 = bundle(&p, &q, &[140000], &[vec![(100, 2.0)], vec![(140000, 2.0)]]);
    assert!(verify_bundle(&Echo, &p, &q, &b).is_ok());
    assert_eq!(verify_bundle(&Echo, &p, &q, &b2).err(), Some(Reject::Prediction));
    // +0 / -0 tie.
    let b = bundle(&p, &q, &[100], &[vec![(100, -0.0)], vec![(140000, 0.0)]]);
    assert!(verify_bundle(&Echo, &p, &q, &b).is_ok());
}

#[test]
fn roster_mutations_reject() {
    let (p, q, b) = two_shard_case();
    let mut x = two_shard_case().2;
    x.children.pop();
    assert!(matches!(verify_bundle(&Echo, &p, &q, &x), Err(Reject::Roster(_))));
    let mut x = two_shard_case().2;
    x.children.remove(5);
    assert!(matches!(verify_bundle(&Echo, &p, &q, &x), Err(Reject::Roster(_))));
    let mut x = two_shard_case().2;
    x.children.swap(3, 4);
    assert!(matches!(verify_bundle(&Echo, &p, &q, &x), Err(Reject::Roster(_))));
    let mut x = two_shard_case().2;
    x.children.swap(37, 38);
    assert!(matches!(verify_bundle(&Echo, &p, &q, &x), Err(Reject::Roster(_))));
    // Duplicate a layer in place of another.
    let mut x = two_shard_case().2;
    let dup = Child {
        receipt: match &x.children[3].receipt {
            ChildReceipt::Stage(r) => ChildReceipt::Stage(r.clone()),
            _ => unreachable!(),
        },
        proof: x.children[3].proof.clone(),
        proof_file: "dup".into(),
    };
    x.children[4] = dup;
    assert!(matches!(verify_bundle(&Echo, &p, &q, &x), Err(Reject::Roster(_))));
    drop(b);
}

#[test]
fn proof_and_receipt_mutations_reject() {
    let (p, q, _) = two_shard_case();
    // A proof that does not verify.
    let mut x = two_shard_case().2;
    x.children[10].proof[5] ^= 1;
    assert!(matches!(verify_bundle(&Echo, &p, &q, &x), Err(Reject::Receipt(10, _))));
    let mut x = two_shard_case().2;
    x.children[10].proof.push(0);
    assert!(matches!(verify_bundle(&Echo, &p, &q, &x), Err(Reject::Proof(10, _))));
    // A layer proof presented as the embedding (role-bound key).
    let mut x = two_shard_case().2;
    x.children[0].proof[0] = 1;
    assert!(matches!(verify_bundle(&Echo, &p, &q, &x), Err(Reject::Proof(0, _))));
    // An altered receipt field without a matching proof.
    let mut x = two_shard_case().2;
    if let ChildReceipt::Stage(r) = &mut x.children[17].receipt {
        r.output[0] ^= 1;
    }
    assert!(matches!(verify_bundle(&Echo, &p, &q, &x), Err(Reject::Receipt(17, _))));
    // An altered (re-proven) boundary breaks the next link.
    let mut x = two_shard_case().2;
    if let ChildReceipt::Stage(r) = &mut x.children[17].receipt {
        r.output[0] ^= 1;
    }
    resign(&p, &q, &mut x.children[17]);
    assert!(matches!(verify_bundle(&Echo, &p, &q, &x), Err(Reject::Receipt(18, _))));
    // A shard whose input is not A36.
    let mut x = two_shard_case().2;
    if let ChildReceipt::Shard(s) = &mut x.children[38].receipt {
        s.input = root(35);
    }
    resign(&p, &q, &mut x.children[38]);
    assert!(matches!(verify_bundle(&Echo, &p, &q, &x), Err(Reject::Receipt(38, _))));
    // Wrong weights per role (0 embed, 20 layer 19, 37 shard 0, 38 shard 1).
    for i in [0usize, 20, 37, 38] {
        let mut x = two_shard_case().2;
        match &mut x.children[i].receipt {
            ChildReceipt::Stage(r) => r.weights[0] ^= 1,
            ChildReceipt::Shard(s) => s.weights[0] ^= 1,
        }
        resign(&p, &q, &mut x.children[i]);
        assert_eq!(verify_bundle(&Echo, &p, &q, &x).err(), Some(Reject::Weights(i)));
    }
    // Positions, window, shard size, winner outside its shard, missing prediction.
    let edits: Vec<Box<dyn Fn(&mut Child)>> = vec![
        Box::new(|c| if let ChildReceipt::Stage(r) = &mut c.receipt { r.positions += 1 }),
        Box::new(|c| if let ChildReceipt::Stage(r) = &mut c.receipt { r.first_prediction += 1 }),
        Box::new(|c| if let ChildReceipt::Stage(r) = &mut c.receipt { r.predictions -= 1 }),
        Box::new(|c| if let ChildReceipt::Stage(r) = &mut c.receipt { (r.first_prediction, r.predictions) = (0, 0) }),
        Box::new(|c| if let ChildReceipt::Shard(s) = &mut c.receipt { s.first_prediction += 1 }),
        Box::new(|c| if let ChildReceipt::Shard(s) = &mut c.receipt { s.predictions -= 1 }),
        Box::new(|c| if let ChildReceipt::Shard(s) = &mut c.receipt { s.shard_log2 = 16 }),
        Box::new(|c| if let ChildReceipt::Shard(s) = &mut c.receipt { s.best[0].0 = 5 }),
        Box::new(|c| if let ChildReceipt::Shard(s) = &mut c.receipt { s.best.pop(); }),
    ];
    for (n, e) in edits.iter().enumerate() {
        let i = if n < 4 { 1 + 18 * (n % 2) } else { 38 };
        let mut x = two_shard_case().2;
        e(&mut x.children[i]);
        resign(&p, &q, &mut x.children[i]);
        let got = verify_bundle(&Echo, &p, &q, &x);
        assert!(matches!(got, Err(Reject::Receipt(_, _))), "edit {n}: {got:?}");
    }
}

#[test]
fn request_and_termination_rules() {
    let (p, _, b) = two_shard_case();
    // A bundle made for another request (nonce) does not verify for this one.
    let other = ClientRequest { nonce: [8; 32], ..req(&[12522, 11], 3) };
    assert!(matches!(verify_bundle(&Echo, &p, &other, &b), Err(Reject::Receipt(0, _))));
    // Different prompt: consumed root differs.
    let other = req(&[12522, 12], 3);
    assert!(verify_bundle(&Echo, &p, &other, &b).is_err());
    // A larger max_reply makes the stopped reply still valid, but a non-stopped short reply is not.
    let q4 = req(&[1], 4);
    let short = bundle(&p, &q4, &[3, 4], &[vec![(3, 1.0), (4, 1.0)], vec![(131072, 0.0), (131073, 0.0)]]);
    assert_eq!(verify_bundle(&Echo, &p, &q4, &short).err(), Some(Reject::Termination));
    // A stop token before the end (the reply continues past the first stop).
    let q3 = req(&[1], 3);
    let early = bundle(&p, &q3, &[151645, 4, 5], &[
        vec![(4, 1.0), (4, 9.0), (5, 9.0)],
        vec![(151645, 9.0), (131073, 0.0), (131073, 0.0)],
    ]);
    assert_eq!(verify_bundle(&Echo, &p, &q3, &early).err(), Some(Reject::Termination));
    // Reply longer than max_reply; empty reply; out-of-vocabulary ID; context overflow.
    let q1 = req(&[1], 1);
    let long = bundle(&p, &q1, &[3, 4], &[vec![(3, 1.0), (4, 1.0)], vec![(131072, 0.0), (131073, 0.0)]]);
    assert_eq!(verify_bundle(&Echo, &p, &q1, &long).err(), Some(Reject::Length));
    let mut empty = bundle(&p, &q1, &[3], &[vec![(3, 1.0)], vec![(131072, 0.0)]]);
    empty.reply.clear();
    assert_eq!(verify_bundle(&Echo, &p, &q1, &empty).err(), Some(Reject::Length));
    let oov = req(&[151936], 3);
    assert_eq!(verify_bundle(&Echo, &p, &oov, &b).err(), Some(Reject::Token));
    let huge = req(&[1], 40960);
    assert_eq!(verify_bundle(&Echo, &p, &huge, &b).err(), Some(Reject::Length));
    // The predicted greedy token differs from the reply.
    let (p, q, mut x) = two_shard_case();
    x.reply[1] = 131072;
    // Built honestly except that the reply's second token is not the greedy one.
    let lie = bundle(&p, &q, &[42, 131072, 151645], &[
        vec![(42, 5.0), (7, 1.0), (9, 0.5)],
        vec![(131072, 4.0), (131500, 2.0), (151645, 3.0)],
    ]);
    assert_eq!(verify_bundle(&Echo, &p, &q, &lie).err(), Some(Reject::Prediction));
    // A NaN winner rejects.
    let nan = bundle(&p, &q, &[42, 131500, 151645], &[
        vec![(42, f32::NAN), (7, 1.0), (9, 0.5)],
        vec![(131072, 4.0), (131500, 2.0), (151645, 3.0)],
    ]);
    assert_eq!(verify_bundle(&Echo, &p, &q, &nan).err(), Some(Reject::Prediction));
    // Policy: unsorted stop IDs, unsupported shard size.
    let mut bad = p.clone();
    bad.stop_ids = vec![151645, 151643];
    assert_eq!(verify_bundle(&Echo, &bad, &q, &x).err(), Some(Reject::Policy));
    let mut bad = p.clone();
    bad.shard_log2 = 2;
    assert_eq!(verify_bundle(&Echo, &bad, &q, &x).err(), Some(Reject::Policy));
}

#[test]
fn release_json_roundtrip_is_strict() {
    let mut p = pins(9);
    p.embed_programs = vec![[1, 2, 3, 4]];
    let v = p.to_json();
    let back = Pins::from_json(&v).unwrap();
    assert_eq!(back.to_json(), v);
    // A derived value that disagrees with the pins rejects.
    let mut bad = v.clone();
    bad["derived"]["suite"] = serde_json::json!("00".repeat(32));
    assert!(Pins::from_json(&bad).is_err());
    // An unknown key rejects.
    let mut bad = v.clone();
    bad["extra"] = serde_json::json!(1);
    assert!(Pins::from_json(&bad).is_err());
    assert_ne!(suite_id(9), suite_id(10));
    let _ = head::shard_count(9);
}

#[test]
fn accumulator_binds_order_count_and_shape() {
    use pci_client::accumulator::{fold, leaf, node};
    let d: Vec<Hash> = (0..334u32).map(root).collect();
    let (n, acc) = fold(&d);
    assert_eq!(n, 334);
    // Canonical shape: first 256 leaves, then 78.
    assert_eq!(node(&fold(&d[..256]), &fold(&d[256..])), (n, acc));
    // Any reordering, omission, duplication or alternative split changes it.
    let mut swapped = d.clone();
    swapped.swap(36, 37);
    assert_ne!(fold(&swapped).1, acc);
    assert_ne!(fold(&d[..333]), (n, acc));
    let mut dup = d.clone();
    dup[200] = dup[199];
    assert_ne!(fold(&dup).1, acc);
    assert_ne!(node(&fold(&d[..128]), &fold(&d[128..])).1, acc);
    // A leaf is not confusable with a one-leaf node chain.
    assert_ne!(leaf(&d[0]), node(&leaf(&d[0]), &leaf(&d[0])));
}
