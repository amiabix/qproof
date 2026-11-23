//! Bound layer guest, single-position baseline: the same canonical kernel, called once
//! per projection per position through `execute_with`. Commits the receipt digest.
#![no_main]
ziskos::entrypoint!(main);

fn main() {
    let record = ziskos::io::read_slice();
    let mut k = pci_dotf8::Kernel::new(pci_f8_guests::Syscall);
    let outcome = pci_layer::execute_with(&record, pci_f8_guests::sha, &mut |x, m, o| {
        k.project_group_bf16(x, x.len(), &[m], &mut [o])
    })
    .expect("layer record rejected");
    ziskos::io::commit_slice(&pci_layer::digest_of(&record, &outcome.receipt));
}
