//! Cost probe, not a relation: `execute_batched` with a null projection kernel (zeros).
//! Everything except the seven projections runs as in `layer-batched`: record checks,
//! in-guest SHA-256 of all 11 tensors and both activations, norms, RoPE, attention,
//! residuals, SiLU, receipt digest. The committed digest is NOT the layer's.
#![no_main]
ziskos::entrypoint!(main);

fn main() {
    let record = ziskos::io::read_slice();
    let outcome = pci_layer::execute_batched(&record, pci_f8_guests::sha, &mut |_x, _cols, _m, o| {
        for out in o.iter_mut() {
            out.fill(0.0);
        }
    })
    .expect("layer record rejected");
    ziskos::io::commit_slice(&pci_layer::digest_of(&record, &outcome.receipt));
}
