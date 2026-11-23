use pci_receipts::*;

struct FixtureBackend;
struct FixtureProof(Role, Hash);
impl ProofBackend for FixtureBackend {
    type Proof = FixtureProof;
    fn verify_child(&self, _: &Hash, role: Role, proof: &FixtureProof) -> Result<Hash, Error> {
        if proof.0 != role {
            return Err(Error::Proof);
        }
        Ok(proof.1)
    }
}
fn release() -> Release {
    Release {
        checkpoint: [1; 32],
        weights: (0..38u8).map(|i| [0x80 | i; 32]).collect(),
        suite: [2; 32],
        context: 40960,
        stop_ids: vec![151645],
    }
}
fn request(prompt: &[u32], max_reply: u32) -> Request<'_> {
    Request {
        nonce: [3; 32],
        prompt,
        max_reply,
    }
}
fn fixtures(r: &Release, q: &Request, reply: &[u32]) -> Vec<(Receipt, FixtureProof)> {
    let mut input = consumed_root(q.prompt, &reply[..reply.len() - 1]);
    let mut children = Vec::new();
    for (i, role) in std::iter::once(Role::Embed)
        .chain((0..36).map(Role::Layer))
        .chain([Role::Head])
        .enumerate()
    {
        let head = role == Role::Head;
        let receipt = Receipt {
            role,
            positions: (q.prompt.len() + reply.len() - 1) as u32,
            first_prediction: if head { q.prompt.len() as u32 - 1 } else { 0 },
            predictions: if head { reply.len() as u32 } else { 0 },
            input,
            output: if head {
                reply_root(reply)
            } else {
                [i as u8; 32]
            },
            weights: r.weights[i],
        };
        input = receipt.output;
        let proof = FixtureProof(role, receipt.digest(r.model(), r.suite, request_id(r, q)));
        children.push((receipt, proof));
    }
    children
}
#[test]
fn complete_minimum_and_target_shapes() {
    for (p, n) in [(1, 1), (500, 250)] {
        let prompt = vec![123; p];
        let reply = vec![456; n];
        let r = release();
        let q = request(&prompt, n as u32);
        let c = fixtures(&r, &q, &reply);
        assert_eq!(
            verify_bundle(&FixtureBackend, &r, &q, &reply, &c)
                .unwrap()
                .end,
            End::Length
        );
    }
}
#[test]
fn roster_proof_role_and_nonce() {
    let r = release();
    let mut q = request(&[1], 1);
    let mut c = fixtures(&r, &q, &[2]);
    assert!(matches!(
        verify_bundle(&FixtureBackend, &r, &q, &[2], &c[..37]),
        Err(Error::Roster)
    ));
    c.swap(1, 2);
    assert!(matches!(
        verify_bundle(&FixtureBackend, &r, &q, &[2], &c),
        Err(Error::Roster)
    ));
    c.swap(1, 2);
    c[1].1.0 = Role::Head;
    assert!(matches!(
        verify_bundle(&FixtureBackend, &r, &q, &[2], &c),
        Err(Error::Proof)
    ));
    c[1].1.0 = Role::Layer(0);
    q.nonce[0] ^= 1;
    assert!(matches!(
        verify_bundle(&FixtureBackend, &r, &q, &[2], &c),
        Err(Error::Receipt)
    ));
}
#[test]
fn individually_authenticated_but_disconnected_or_wrong_window() {
    let r = release();
    let q = request(&[1, 2], 2);
    for (child, field) in [
        (0, "input"),
        (18, "input"),
        (37, "output"),
        (37, "window"),
        (2, "positions"),
    ] {
        let mut c = fixtures(&r, &q, &[3, 4]);
        let (rec, proof) = &mut c[child];
        match field {
            "input" => rec.input[0] ^= 1,
            "output" => rec.output[0] ^= 1,
            "window" => rec.first_prediction += 1,
            _ => rec.positions += 1,
        }
        proof.1 = rec.digest(r.model(), r.suite, request_id(&r, &q));
        assert!(matches!(
            verify_bundle(&FixtureBackend, &r, &q, &[3, 4], &c),
            Err(Error::Receipt)
        ));
    }
}
#[test]
fn transcript_and_stopping() {
    let r = release();
    let q = request(&[7], 2);
    let stop = 151645;
    let c = fixtures(&r, &q, &[9, stop]);
    assert_eq!(
        verify_bundle(&FixtureBackend, &r, &q, &[9, stop], &c)
            .unwrap()
            .end,
        End::Stop
    );
    assert!(matches!(
        verify_bundle(&FixtureBackend, &r, &q, &[stop, 9], &c),
        Err(Error::Termination)
    ));
    assert!(matches!(
        verify_bundle(&FixtureBackend, &r, &q, &[9], &c),
        Err(Error::Termination)
    ));
    assert!(matches!(
        verify_bundle(&FixtureBackend, &r, &q, &[10, stop], &c),
        Err(Error::Receipt)
    ));
    assert!(matches!(
        verify_bundle(&FixtureBackend, &r, &q, &[9, 151936], &c),
        Err(Error::Token)
    ));
    assert!(matches!(
        verify_bundle(&FixtureBackend, &r, &request(&[7], u32::MAX), &[9], &c),
        Err(Error::Length)
    ));
}

#[test]
fn activation_commitment_binds_boundary_request_shape_and_signed_zero() {
    let mut data = vec![0.0; 2560];
    let hash = activation_root([1; 32], [2; 32], 0, 1, &data).unwrap();
    assert_ne!(
        hash,
        activation_root([1; 32], [2; 32], 1, 1, &data).unwrap()
    );
    assert_ne!(
        hash,
        activation_root([1; 32], [3; 32], 0, 1, &data).unwrap()
    );
    assert!(activation_root([1; 32], [2; 32], 0, 2, &data).is_err());
    data[17] = -0.0;
    assert_ne!(
        hash,
        activation_root([1; 32], [2; 32], 0, 1, &data).unwrap()
    );
}

#[test]
fn rejects_unsupported_or_ambiguous_release_policy() {
    let mut r = release();
    let q = request(&[1], 1);
    let c = fixtures(&r, &q, &[2]);
    r.context = 40961;
    assert!(matches!(
        verify_bundle(&FixtureBackend, &r, &q, &[2], &c),
        Err(Error::Policy)
    ));
    r.context = 40960;
    r.stop_ids.push(r.stop_ids[0]);
    assert!(matches!(
        verify_bundle(&FixtureBackend, &r, &q, &[2], &c),
        Err(Error::Policy)
    ));
}

#[test]
fn every_stage_rejects_a_broken_input_output_or_position() {
    let r = release();
    let q = request(&[11, 22], 3);
    let reply = [7, 8, 9];
    for child in 0..38 {
        for field in 0..5 {
            let mut c = fixtures(&r, &q, &reply);
            let (receipt, proof) = &mut c[child];
            match field {
                0 => receipt.input[child % 32] ^= 1,
                1 => receipt.output[child % 32] ^= 1,
                2 => receipt.positions += 1,
                3 => receipt.first_prediction += 1,
                _ => receipt.predictions += 1,
            }
            // Each altered receipt passes the fixture backend individually.
            // Independent of digest authentication, the join must reject it.
            proof.1 = receipt.digest(r.model(), r.suite, request_id(&r, &q));
            assert!(
                matches!(
                    verify_bundle(&FixtureBackend, &r, &q, &reply, &c),
                    Err(Error::Receipt)
                ),
                "{child} {field}"
            );
        }
    }
}

#[test]
fn every_stage_rejects_bad_proof_or_shifted_roster() {
    let r = release();
    let q = request(&[9], 1);
    for child in 0..38 {
        let mut c = fixtures(&r, &q, &[3]);
        c[child].1.1[0] ^= 1;
        assert!(matches!(
            verify_bundle(&FixtureBackend, &r, &q, &[3], &c),
            Err(Error::Receipt)
        ));
        c[child].1.1[0] ^= 1;
        c.swap(child, (child + 1) % 38);
        assert!(matches!(
            verify_bundle(&FixtureBackend, &r, &q, &[3], &c),
            Err(Error::Roster)
        ));
    }
}

#[test]
fn request_replay_and_context_boundaries() {
    let mut r = release();
    let prompt = vec![1; 40959];
    let q = request(&prompt, 1);
    let c = fixtures(&r, &q, &[2]);
    assert!(verify_bundle(&FixtureBackend, &r, &q, &[2], &c).is_ok());
    assert!(matches!(
        verify_bundle(&FixtureBackend, &r, &request(&prompt, 2), &[2], &c),
        Err(Error::Length)
    ));
    for changed in [request(&[], 1), request(&[1], 0)] {
        assert!(matches!(
            verify_bundle(&FixtureBackend, &r, &changed, &[2], &c),
            Err(Error::Length)
        ));
    }
    let q = request(&[7], 1);
    let c = fixtures(&r, &q, &[2]);
    for changed in [request(&[8], 1), request(&[7, 8], 1)] {
        assert!(matches!(
            verify_bundle(&FixtureBackend, &r, &changed, &[2], &c),
            Err(Error::Receipt)
        ));
    }
    r.checkpoint[0] ^= 1;
    assert!(matches!(
        verify_bundle(&FixtureBackend, &r, &q, &[2], &c),
        Err(Error::Receipt)
    ));
    r.checkpoint[0] ^= 1;
    r.suite[0] ^= 1;
    assert!(matches!(
        verify_bundle(&FixtureBackend, &r, &q, &[2], &c),
        Err(Error::Receipt)
    ));
}

#[test]
fn every_stage_rejects_weights_other_than_the_pinned_enrollment() {
    let r = release();
    let q = request(&[5], 1);
    for child in 0..38 {
        // A child that honestly hashed different weights: digest matches its receipt.
        let mut c = fixtures(&r, &q, &[6]);
        c[child].0.weights[3] ^= 1;
        c[child].1.1 = c[child].0.digest(r.model(), r.suite, request_id(&r, &q));
        assert_eq!(
            verify_bundle(&FixtureBackend, &r, &q, &[6], &c).err(),
            Some(Error::Weights)
        );
        // Changing one pinned stage root changes ModelID, so every digest fails.
        let mut pinned = release();
        pinned.weights[child][0] ^= 1;
        let c = fixtures(&r, &q, &[6]);
        assert!(verify_bundle(&FixtureBackend, &pinned, &q, &[6], &c).is_err());
    }
    let mut short = release();
    short.weights.pop();
    assert_eq!(
        verify_bundle(&FixtureBackend, &short, &q, &[6], &fixtures(&r, &q, &[6])).err(),
        Some(Error::Policy)
    );
}

#[test]
fn stage_roster_matches_enrollment_order() {
    let names = stage_tensors(Role::Layer(7));
    let mut sorted = names.clone();
    sorted.sort();
    assert_eq!(names, sorted);
    assert_eq!(names.len(), 11);
    assert_eq!(names[0], "model.layers.7.input_layernorm.weight");
    assert_ne!(
        weights_root(Role::Layer(0), &[[0; 32]; 11]),
        weights_root(Role::Layer(1), &[[0; 32]; 11])
    );
}
