//! exporter KIND RECORD CAPTURE EXPECTED_DIGEST (64 hex digits or a raw digest file).
use sha2::{Digest, Sha256};
use std::{env, error::Error, fs, path::Path};

type Result<T> = std::result::Result<T, Box<dyn Error>>;

fn unframe(bytes: &[u8]) -> Result<&[u8]> {
    let prefix = bytes.get(..8).ok_or("missing record length")?;
    let size = usize::try_from(u64::from_le_bytes(prefix.try_into()?))?;
    let payload = &bytes[8..];
    let record = payload.get(..size).ok_or("truncated record")?;
    let padding = &payload[size..];
    if padding.len() != size.wrapping_neg() % 8 || padding.iter().any(|&b| b != 0) {
        return Err("incorrect record padding or trailing data".into());
    }
    Ok(record)
}

fn expected_digest(arg: &str) -> Result<[u8; 32]> {
    if arg.len() == 64 && arg.bytes().all(|b| b.is_ascii_hexdigit()) {
        let mut digest = [0; 32];
        for (i, byte) in digest.iter_mut().enumerate() {
            *byte = u8::from_str_radix(&arg[2 * i..2 * i + 2], 16)?;
        }
        Ok(digest)
    } else {
        fs::read(arg)?
            .try_into()
            .map_err(|_| "expected a 32-byte digest file".into())
    }
}

// Qproof passes matrix slices borrowed directly from the unframed record.
// Both x and out are position-major. The kernel has already rounded out to BF16.
fn append_matrix(
    buf: &mut Vec<u8>,
    record: &[u8],
    x: &[u32],
    k: usize,
    matrix: &[u8],
    out: &[f32],
) {
    let offset = matrix.as_ptr() as usize - record.as_ptr() as usize;
    assert!(matrix.len() <= record.len() && offset <= record.len() - matrix.len());
    let t = x.len() / k;
    let r = matrix.len() / (2 * k);
    assert_eq!(
        (t * k, r * 2 * k, t * r),
        (x.len(), matrix.len(), out.len())
    );
    for word in [offset, t, r, k] {
        buf.extend_from_slice(&(word as u64).to_le_bytes());
    }
    for bits in x {
        buf.extend_from_slice(&bits.to_le_bytes());
    }
    for value in out {
        buf.extend_from_slice(&value.to_le_bytes());
    }
}

fn run() -> Result<()> {
    let args: Vec<_> = env::args().skip(1).collect();
    let [kind, input, capture, expected] = args.as_slice() else {
        return Err("usage: exporter layer|head RECORD CAPTURE EXPECTED_DIGEST".into());
    };
    let expected = expected_digest(expected)?;
    let framed = fs::read(input)?;
    let record = unframe(&framed)?;
    let sha = |bytes: &[u8]| Sha256::digest(bytes).into();
    let mut buf = Vec::new();
    let digest = match kind.as_str() {
        "layer" => {
            let outcome = pci_layer::execute_batched(record, sha, &mut |x, k, matrices, outs| {
                pci_layer::project_batch_bf16(x, k, matrices, outs);
                for (matrix, out) in matrices.iter().zip(outs) {
                    append_matrix(&mut buf, record, x, k, matrix, out);
                }
            })?;
            pci_layer::digest_of(record, &outcome.receipt)
        }
        "head" => {
            pci_stages::head::execute_with(record, sha, &mut |x, matrix, out| {
                pci_stages::head::reference_kernel(x, matrix, out);
                append_matrix(&mut buf, record, x, pci_stages::DIM as usize, matrix, out);
            })?
            .digest
        }
        _ => return Err("KIND must be layer or head".into()),
    };
    if digest != expected {
        return Err(format!(
            "receipt digest mismatch: computed {:x}",
            sha2::digest::Output::<Sha256>::from(digest)
        )
        .into());
    }
    fs::write(Path::new(capture), buf)?;
    Ok(())
}

fn main() {
    if let Err(err) = run() {
        eprintln!("exporter: {err}");
        std::process::exit(1);
    }
}
