//! Layer child guest: authenticates the record, computes the layer and
//! commits only the 32-byte receipt digest (ABI v1, zero padded by ZisK).
#![no_main]
ziskos::entrypoint!(main);

fn sha(bytes: &[u8]) -> [u8; 32] {
    ziskos::zisklib::sha256(bytes)
}

fn main() {
    let record = ziskos::io::read_slice();
    let outcome = pci_layer::execute(&record, sha).expect("layer record rejected");
    ziskos::io::commit_slice(&pci_layer::digest_of(&record, &outcome.receipt));
}
