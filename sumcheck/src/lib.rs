//! The shared parts of the matmul proof: hashes, transcript, proof format and
//! verifier, used by the native prover and the ZisK guest.

use ark_bn254::Fr;
use ark_crypto_primitives::crh::{CRHScheme, TwoToOneCRHScheme};
use ark_crypto_primitives::merkle_tree::{ByteDigestConverter, Config};
use ark_crypto_primitives::sponge::{Absorb, CryptographicSponge};
use ark_ff::{BigInteger, Field, One, PrimeField, Zero};
use ark_poly::DenseMultilinearExtension;
use ark_poly_commit::linear_codes::{LigeroPCParams, LinearCodePCS, MultilinearLigero};
use ark_poly_commit::{LabeledCommitment, PolynomialCommitment};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use ark_std::borrow::Borrow;
use ark_std::rand::Rng;

pub type CrhResult<T> = std::result::Result<T, ark_crypto_primitives::Error>;

// Blake3 for the column hashes, the Merkle tree and the transcript.

pub struct Blake3;

impl TwoToOneCRHScheme for Blake3 {
    type Input = [u8];
    type Output = Vec<u8>;
    type Parameters = ();
    fn setup<R: Rng>(_: &mut R) -> CrhResult<()> {
        Ok(())
    }
    fn evaluate<T: Borrow<[u8]>>(_: &(), l: T, r: T) -> CrhResult<Vec<u8>> {
        let mut h = blake3::Hasher::new();
        h.update(l.borrow());
        h.update(r.borrow());
        Ok(h.finalize().as_bytes().to_vec())
    }
    fn compress<T: Borrow<Vec<u8>>>(p: &(), l: T, r: T) -> CrhResult<Vec<u8>> {
        <Self as TwoToOneCRHScheme>::evaluate(p, l.borrow().as_slice(), r.borrow().as_slice())
    }
}

/// Leaves are already column digests.
pub struct Identity;

impl CRHScheme for Identity {
    type Input = Vec<u8>;
    type Output = Vec<u8>;
    type Parameters = ();
    fn setup<R: Rng>(_: &mut R) -> CrhResult<()> {
        Ok(())
    }
    fn evaluate<T: Borrow<Vec<u8>>>(_: &(), input: T) -> CrhResult<Vec<u8>> {
        Ok(input.borrow().clone())
    }
}

/// Hashes a column of field elements with Blake3.
pub struct ColHash;

impl CRHScheme for ColHash {
    type Input = Vec<Fr>;
    type Output = Vec<u8>;
    type Parameters = ();
    fn setup<R: Rng>(_: &mut R) -> CrhResult<()> {
        Ok(())
    }
    fn evaluate<T: Borrow<Vec<Fr>>>(_: &(), input: T) -> CrhResult<Vec<u8>> {
        let mut h = blake3::Hasher::new();
        for f in input.borrow() {
            h.update(&f.into_bigint().to_bytes_le());
        }
        Ok(h.finalize().as_bytes().to_vec())
    }
}

pub struct Mt;

impl Config for Mt {
    type Leaf = Vec<u8>;
    type LeafDigest = Vec<u8>;
    type LeafInnerDigestConverter = ByteDigestConverter<Vec<u8>>;
    type InnerDigest = Vec<u8>;
    type LeafHash = Identity;
    type TwoToOneHash = Blake3;
}

pub type Poly = DenseMultilinearExtension<Fr>;
pub type Pcs = LinearCodePCS<MultilinearLigero<Fr, Mt, Poly, ColHash>, Fr, Poly, Mt, ColHash>;
pub type Com = LabeledCommitment<<Pcs as PolynomialCommitment<Fr, Poly>>::Commitment>;

/// A Blake3 Fiat-Shamir transcript.
#[derive(Clone)]
pub struct Transcript(blake3::Hasher);

impl CryptographicSponge for Transcript {
    type Config = ();
    fn new(_: &()) -> Self {
        Transcript(blake3::Hasher::new())
    }
    fn absorb(&mut self, input: &impl Absorb) {
        self.0.update(&input.to_sponge_bytes_as_vec());
    }
    fn squeeze_bytes(&mut self, n: usize) -> Vec<u8> {
        let mut out = vec![0u8; n];
        self.0.finalize_xof().fill(&mut out);
        self.0.update(b"squeeze");
        self.0.update(&out);
        out
    }
    fn squeeze_bits(&mut self, n: usize) -> Vec<bool> {
        let bytes = self.squeeze_bytes(n.div_ceil(8));
        (0..n).map(|i| bytes[i / 8] >> (i % 8) & 1 == 1).collect()
    }
}

pub fn challenge(t: &mut Transcript) -> Fr {
    t.squeeze_field_elements::<Fr>(1)[0]
}

/// `eq(r, i)` for every `i`, variable `k` being bit `k` of `i`.
pub fn eq_table(r: &[Fr]) -> Vec<Fr> {
    let mut t = vec![Fr::one()];
    for &rk in r {
        let mut n = vec![Fr::zero(); t.len() * 2];
        for (i, &v) in t.iter().enumerate() {
            n[i] = v * (Fr::one() - rk);
            n[i + t.len()] = v * rk;
        }
        t = n;
    }
    t
}

pub fn mle_eval(v: &[Fr], r: &[Fr]) -> Fr {
    eq_table(r).iter().zip(v).map(|(a, b)| *a * b).sum()
}

#[derive(CanonicalSerialize, CanonicalDeserialize)]
pub struct Proof {
    /// Per round: g(0), g(1), g(2).
    pub rounds: Vec<[Fr; 3]>,
    pub value: Fr,
    pub opening: <Pcs as PolynomialCommitment<Fr, Poly>>::Proof,
}

impl Proof {
    pub fn bytes(&self) -> usize {
        self.rounds.len() * 3 * 32 + 32 + self.opening.compressed_size()
    }
}

pub fn absorb_public(t: &mut Transcript, com: &Com, xs: &[Vec<Fr>], ys: &[Vec<Fr>]) {
    let mut bytes = Vec::new();
    com.commitment().serialize_compressed(&mut bytes).unwrap();
    t.absorb(&bytes);
    for v in xs.iter().chain(ys) {
        t.absorb(v);
    }
}

pub fn combine(vs: &[Vec<Fr>], rho: Fr) -> Vec<Fr> {
    let mut out = vec![Fr::zero(); vs[0].len()];
    let mut p = Fr::one();
    for v in vs {
        for (o, &x) in out.iter_mut().zip(v) {
            *o += p * x;
        }
        p *= rho;
    }
    out
}

pub fn verify(
    m: &Dims,
    vk: &<Pcs as PolynomialCommitment<Fr, Poly>>::VerifierKey,
    com: &Com,
    xs: &[Vec<Fr>],
    ys: &[Vec<Fr>],
    p: &Proof,
    t: &mut Transcript,
) -> bool {
    absorb_public(t, com, xs, ys);
    let r: Vec<Fr> = (0..m.row_vars as usize).map(|_| challenge(t)).collect();
    let rho = challenge(t);
    let mut claim = mle_eval(&combine(ys, rho), &r);
    let (two, inv2) = (Fr::from(2u64), Fr::from(2u64).inverse().unwrap());
    let mut s = Vec::new();
    for g in &p.rounds {
        if g[0] + g[1] != claim {
            return false;
        }
        t.absorb(&g.to_vec());
        let c = challenge(t);
        // The degree-2 polynomial through g(0), g(1), g(2), at c.
        claim = g[0] * (c - Fr::one()) * (c - two) * inv2 - g[1] * c * (c - two)
            + g[2] * c * (c - Fr::one()) * inv2;
        s.push(c);
    }
    if p.value * mle_eval(&combine(xs, rho), &s) != claim {
        return false;
    }
    let point: Vec<Fr> = s.iter().chain(&r).copied().collect();
    Pcs::check(vk, [com], &point, [p.value], &p.opening, t, None).unwrap()
}

/// A matrix's shape: rows and columns, and their padded variable counts.
#[derive(Clone, Copy, CanonicalSerialize, CanonicalDeserialize)]
pub struct Dims {
    pub rows: u64,
    pub cols: u64,
    pub row_vars: u64,
    pub col_vars: u64,
}

/// Ligero over BN254 with blowup 2 and a 128-bit target, Blake3 throughout.
pub fn keys() -> (
    <Pcs as PolynomialCommitment<Fr, Poly>>::CommitterKey,
    <Pcs as PolynomialCommitment<Fr, Poly>>::VerifierKey,
) {
    let pp = LigeroPCParams::<Fr, Mt, ColHash>::new(128, 2, true, (), (), ());
    Pcs::trim(&pp, 0, 0, None).unwrap()
}

/// One matrix's statement and proof: W committed, x and y public.
#[derive(CanonicalSerialize, CanonicalDeserialize)]
pub struct MatrixProof {
    pub dims: Dims,
    pub commitment: <Pcs as PolynomialCommitment<Fr, Poly>>::Commitment,
    pub xs: Vec<Vec<Fr>>,
    pub ys: Vec<Vec<Fr>>,
    pub proof: Proof,
}

impl MatrixProof {
    pub fn verify(&self, vk: &<Pcs as PolynomialCommitment<Fr, Poly>>::VerifierKey) -> bool {
        let com = LabeledCommitment::new(String::new(), self.commitment.clone(), None);
        verify(
            &self.dims,
            vk,
            &com,
            &self.xs,
            &self.ys,
            &self.proof,
            &mut Transcript::new(&()),
        )
    }
}
