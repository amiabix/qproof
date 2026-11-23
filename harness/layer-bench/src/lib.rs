//! Shared pieces of the native driver and its tests.
use pci_dotf8::{Kernel, SoftDotF8};
use pci_layer::Outcome;
use pci_receipts::Hash;
use sha2::{Digest, Sha256};

pub fn sha(b: &[u8]) -> Hash {
    Sha256::digest(b).into()
}

pub fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}

/// The guest's batched arm with the software `dot_f8`.
pub fn batched(record: &[u8]) -> Result<Outcome, &'static str> {
    let mut k = Kernel::new(SoftDotF8);
    pci_layer::execute_batched(record, sha, &mut |x, cols, m, o| k.project_group_bf16(x, cols, m, o))
}

/// The guest's per-position arm with the software `dot_f8`.
pub fn per_position(record: &[u8]) -> Result<Outcome, &'static str> {
    let mut k = Kernel::new(SoftDotF8);
    pci_layer::execute_with(record, sha, &mut |x, m, o| {
        k.project_group_bf16(x, x.len(), &[m], &mut [o])
    })
}

/// The record inside a ZisK stdin file (u64 length, record, zero padding).
pub fn record_of_stdin(stdin: &[u8]) -> &[u8] {
    let n = u64::from_le_bytes(stdin[..8].try_into().unwrap()) as usize;
    assert!(stdin.len() >= 8 + n && stdin[8 + n..].iter().all(|b| *b == 0), "stdin framing");
    &stdin[8..8 + n]
}

/// ZisK stdin framing of `payload`.
pub fn stdin_of(payload: &[u8]) -> Vec<u8> {
    let mut v = (payload.len() as u64).to_le_bytes().to_vec();
    v.extend_from_slice(payload);
    v.resize(v.len().next_multiple_of(8), 0);
    v
}

/// Projection sample input for the `projection-sample` guest:
/// `"PCISMPL1" | mode u32 | positions u32 | rows u32 | cols u32 | x (positions*cols FP32
/// bits) | matrix (rows*cols raw BF16)`. Mode 0 = reference ordered FP32, 1 = canonical
/// kernel one position per call, 2 = canonical kernel, all positions per call.
/// Modes 3, 4 and 5 exercise the same execution layouts for the BF16 serving profile.
pub fn sample_payload(mode: u32, positions: usize, cols: usize, x: &[u32], matrix: &[u8]) -> Vec<u8> {
    let rows = matrix.len() / (2 * cols);
    let mut v = b"PCISMPL1".to_vec();
    for f in [mode, positions as u32, rows as u32, cols as u32] {
        v.extend(f.to_le_bytes());
    }
    v.extend(x.iter().flat_map(|b| b.to_le_bytes()));
    v.extend_from_slice(matrix);
    v
}
