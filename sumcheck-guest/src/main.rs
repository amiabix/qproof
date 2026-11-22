//! Runs one Qwen3 segment (a layer or the head), taking the matrix products
//! from `qwen-sumcheck` proofs and checking each proof against the inputs
//! computed here, so the prover cannot choose them.
//!
//! Input, two records: the segment input without its matrices, then the
//! `Vec<MatrixProof>` with empty `xs`. Commits 96 bytes: sha256(segment
//! input), `weights_id(proofs)`, sha256(output).

#![no_main]

ziskos::entrypoint!(main);

use std::cell::RefCell;

use ark_bn254::Fr;
use ark_serialize::CanonicalDeserialize;
use qwen_common::{dot_soft, qdot_soft, run_with};
use qwen_sumcheck::{MatrixProof, encode_x, keys, weights_id, y_to_f32};
use ziskos::zisklib::sha256;

fn main() {
    let input = ziskos::io::read_slice();
    let mut proofs =
        Vec::<MatrixProof>::deserialize_uncompressed_unchecked(&ziskos::io::read_slice()[..])
            .expect("proofs");
    let n = proofs.iter().map(|p| p.matrix as usize + 1).max().unwrap();
    // For each matrix: its inputs so far, encoded as in the proofs, and the
    // exponent of the latest one.
    let seen = RefCell::new((vec![Vec::<Vec<Fr>>::new(); n], vec![0i32; n]));
    let chunks = |m: usize| proofs.iter().filter(move |p| p.matrix as usize == m);
    let mut obs = |m: usize, x: &[f32]| {
        let vars = chunks(m)
            .next()
            .expect("no proof for a matrix")
            .dims
            .col_vars;
        let (xe, e0, _) = encode_x(x, vars as usize);
        let (xs, exp) = &mut *seen.borrow_mut();
        xs[m].push(xe);
        exp[m] = e0;
    };
    let mut claimed = |m: usize, out: &mut [f32]| {
        let (xs, exp) = &*seen.borrow();
        let t = xs[m].len() - 1;
        let mut at = 0;
        for p in chunks(m) {
            let rows = p.dims.rows as usize;
            for (i, o) in out[at..at + rows].iter_mut().enumerate() {
                *o = y_to_f32(p.ys[t][i], p.row_exp[i], exp[m]);
            }
            at += rows;
        }
        assert_eq!(at, out.len(), "proofs cover {at} of {} rows", out.len());
    };
    let out = run_with(&input, dot_soft, qdot_soft, &mut obs, Some(&mut claimed));

    let (xs, _) = seen.into_inner();
    let (_, vk) = keys();
    for (k, p) in proofs.iter_mut().enumerate() {
        // Use the inputs computed here, never ones from the prover.
        p.xs = xs[p.matrix as usize].clone();
        assert!(p.verify(&vk), "proof {k} rejected");
    }
    let bytes: Vec<u8> = out.iter().flat_map(|v| v.to_le_bytes()).collect();
    let mut commit = sha256(&input).to_vec();
    commit.extend(weights_id(&proofs));
    commit.extend(sha256(&bytes));
    ziskos::io::commit_slice(&commit);
}
