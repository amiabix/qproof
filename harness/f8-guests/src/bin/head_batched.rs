//! Authenticated vocabulary shard using the same lossless BF16 kernel as the layers.
#![no_main]
ziskos::entrypoint!(main);

fn main() {
    let record = ziskos::io::read_slice();
    let mut kernel = pci_dotf8::Kernel::new(pci_f8_guests::Syscall);
    let outcome = pci_stages::head::execute_with(&record, pci_f8_guests::sha, &mut |x, weights, out| {
        kernel.project_group_bf16(x, pci_stages::DIM as usize, &[weights], &mut [out]);
    })
    .expect("head shard record rejected");
    ziskos::io::commit_slice(&outcome.digest);
}
