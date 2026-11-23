//! Isolated receipt-checking cost. Input digests model the outputs of a
//! separately timed proof verifier; this experimental ELF cannot authenticate proofs.
#![no_main]
use pci_receipts::*;
ziskos::entrypoint!(main);

struct SuppliedDigests;
impl ProofBackend for SuppliedDigests {
    type Proof = Hash;
    fn verify_child(&self, _: &Hash, _: Role, digest: &Hash) -> Result<Hash, Error> {
        Ok(*digest)
    }
}
fn take<const N: usize>(input: &mut &[u8]) -> [u8; N] {
    let (first, rest) = input.split_at(N);
    *input = rest;
    first.try_into().unwrap()
}
fn number(input: &mut &[u8]) -> u32 {
    u32::from_le_bytes(take(input))
}
fn tokens(input: &mut &[u8], n: u32) -> Vec<u32> {
    assert!(n <= 40960);
    (0..n).map(|_| number(input)).collect()
}
fn main() {
    let record = ziskos::io::read_slice();
    let mut input = &record[..];
    assert_eq!(take::<8>(&mut input), *b"JOINTEST");
    let checkpoint = take(&mut input);
    let suite = take(&mut input);
    let weights: Vec<Hash> = (0..38).map(|_| take(&mut input)).collect();
    let r = Release {
        checkpoint,
        weights,
        suite,
        context: 40960,
        stop_ids: vec![151645],
    };
    let nonce = take(&mut input);
    let p = number(&mut input);
    let n = number(&mut input);
    let prompt = tokens(&mut input, p);
    let reply = tokens(&mut input, n);
    let q = Request {
        nonce,
        prompt: &prompt,
        max_reply: n,
    };
    let mut children = Vec::new();
    for _ in 0..38 {
        let kind = number(&mut input);
        let index = number(&mut input);
        let role = match (kind, index) {
            (0, 0) => Role::Embed,
            (1, i) => Role::Layer(i),
            (2, 0) => Role::Head,
            _ => panic!("bad role"),
        };
        children.push((
            Receipt {
                role,
                positions: number(&mut input),
                first_prediction: number(&mut input),
                predictions: number(&mut input),
                input: take(&mut input),
                output: take(&mut input),
                weights: take(&mut input),
            },
            take(&mut input),
        ));
    }
    assert!(input.is_empty());
    let s = verify_bundle(&SuppliedDigests, &r, &q, &reply, &children).unwrap();
    let mut public = b"JOINTEST".to_vec();
    public.extend(s.request_id);
    public.extend(s.reply_root);
    ziskos::io::commit_slice(&public);
}
