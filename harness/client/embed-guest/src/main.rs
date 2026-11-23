//! Embedding child guest: opens every consumed row of the tied BF16 matrix
//! under the row commitment, widens it exactly and commits only the 32-byte
//! receipt digest (ABI v1, zero padded by ZisK).
#![no_main]
ziskos::entrypoint!(main);

fn sha(bytes: &[u8]) -> [u8; 32] {
    ziskos::zisklib::sha256(bytes)
}

fn main() {
    let record = ziskos::io::read_slice();
    let out = pci_stages::embed::execute(&record, sha).expect("embedding record rejected");
    ziskos::io::commit_slice(&out.digest);
}
