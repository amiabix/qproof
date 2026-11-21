//! Verifies one layer's matmul proofs (`qwen-sumcheck` output) and commits
//! sha256 of the input, which holds the commitments and the public x and y.

#![no_main]

ziskos::entrypoint!(main);

use ark_serialize::CanonicalDeserialize;
use qwen_sumcheck::{MatrixProof, keys};

fn main() {
    let input = ziskos::io::read_slice();
    let proofs =
        Vec::<MatrixProof>::deserialize_uncompressed_unchecked(&input[..]).expect("proofs");
    let (_, vk) = keys();
    for (i, p) in proofs.iter().enumerate() {
        assert!(p.verify(&vk), "matrix {i} rejected");
    }
    println!("verified {} matrices", proofs.len());
    ziskos::io::commit_slice(&ziskos::zisklib::sha256(&input));
}
