//! Head shard child guest: authenticates one aligned vocabulary shard of the
//! tied BF16 matrix and the final norm, recomputes the final-RMSNorm logits of
//! every requested position for that shard under the pinned BF16 serving profile and
//! commits only the 32-byte shard receipt digest (ABI v1, zero padded).
#![no_main]
ziskos::entrypoint!(main);

fn sha(bytes: &[u8]) -> [u8; 32] {
    ziskos::zisklib::sha256(bytes)
}

fn main() {
    let record = ziskos::io::read_slice();
    let out = pci_stages::head::execute(&record, sha).expect("head shard record rejected");
    ziskos::io::commit_slice(&out.digest);
}
