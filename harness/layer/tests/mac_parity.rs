//! The layer's reference projections against PyTorch 2.13.0 CPU (arm64) BF16 `linear`
//! on the original checkpoint: layer 0, q/k/v of the first three positions.
//! Fixtures (workspace `target/`): `norm-fixture/layer00.v1.stdin`, `norm-fixture/norm.f32`
//! (the 7680 BF16-valued input-norm outputs) and `ref-{q,k,v}.bin` from the torch oracle.
use pci_layer::*;
use sha2::{Digest, Sha256};

fn words(b: &[u8]) -> Vec<u32> {
    b.chunks_exact(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect()
}

#[test]
#[ignore = "needs the layer-0 checkpoint fixture and torch references in target/"]
fn layer0_qkv_equal_torch_bf16_linear() {
    let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/../../target");
    let read = |p: &str| std::fs::read(format!("{dir}/{p}")).unwrap();
    let stdin = read("norm-fixture/layer00.v1.stdin");
    let n = u64::from_le_bytes(stdin[..8].try_into().unwrap()) as usize;
    // The fixture predates the prediction window in the header; claim R = 1.
    let mut record = stdin[8..8 + n].to_vec();
    record[112 + 44..112 + 48].copy_from_slice(&1u32.to_le_bytes());
    let norm = words(&read("norm-fixture/norm.f32"));
    let refs = [read("ref-q.bin"), read("ref-k.bin"), read("ref-v.bin")].map(|b| words(&b));
    assert_eq!(norm.len(), 7680);

    // Through the layer: the first nine kernel calls are q, k, v of positions 0, 1, 2.
    let mut calls = 0;
    execute_with(&record, |b| Sha256::digest(b).into(), &mut |x, m, out| {
        project_bf16_bits(x, m, out);
        if calls < 9 {
            let (t, p) = (calls / 3, calls % 3);
            assert_eq!(x, &norm[t * 2560..][..2560], "input t {t}");
            let rows = out.len();
            let got: Vec<u32> = out.iter().map(|v| v.to_bits()).collect();
            let want = &refs[p][t * rows..][..rows];
            let diff = got.iter().zip(want).filter(|(a, b)| a != b).count();
            assert_eq!(diff, 0, "projection {p} position {t}: {diff} of {rows} differ");
        }
        calls += 1;
    })
    .unwrap();
    assert_eq!(calls, 21);
}
