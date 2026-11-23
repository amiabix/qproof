//! Bound layer guest, batched arm: every projection sees all positions; q/k/v and
//! gate/up share one prepared input. Commits only the 32-byte receipt digest.
#![no_main]
ziskos::entrypoint!(main);

fn main() {
    let record = ziskos::io::read_slice();
    let mut k = pci_dotf8::Kernel::new(pci_f8_guests::Syscall);
    let outcome = pci_layer::execute_batched(&record, pci_f8_guests::sha, &mut |x, cols, m, o| {
        k.project_group_bf16(x, cols, m, o)
    })
    .expect("layer record rejected");
    ziskos::io::commit_slice(&pci_layer::digest_of(&record, &outcome.receipt));
}
