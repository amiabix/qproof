//! Projection sample benchmark (input: pci_layer_bench::sample_payload). Mode 0: the
//! reference ordered FP32 loop; 1: the canonical kernel, one position per call; 2: the
//! canonical kernel, all positions per call. Modes 3 through 5 use the BF16 serving profile. Commits sha256 of the output FP32 bits.
#![no_main]
ziskos::entrypoint!(main);

fn main() {
    let input = ziskos::io::read_slice();
    assert_eq!(&input[..8], b"PCISMPL1", "magic");
    let u = |i: usize| u32::from_le_bytes(input[8 + 4 * i..12 + 4 * i].try_into().unwrap()) as usize;
    let (mode, n, rows, cols) = (u(0), u(1), u(2), u(3));
    let at = 24;
    assert_eq!(input.len(), at + 4 * n * cols + 2 * rows * cols, "length");
    // SAFETY: read_slice data is 8-byte aligned and `at` is a multiple of 4.
    let x: &[u32] = unsafe { core::slice::from_raw_parts(input[at..].as_ptr().cast(), n * cols) };
    let matrix = &input[at + 4 * n * cols..];
    let mut out = vec![0f32; n * rows];
    let mut k = pci_dotf8::Kernel::new(pci_f8_guests::Syscall);
    match mode {
        0 => pci_dotf8::reference(x, cols, matrix, &mut out),
        1 => {
            for (xt, ot) in x.chunks_exact(cols).zip(out.chunks_exact_mut(rows)) {
                k.project_group(xt, cols, &[matrix], &mut [ot]);
            }
        }
        2 => k.project_group(x, cols, &[matrix], &mut [out.as_mut_slice()]),
        3 => pci_layer::project_batch_bf16(x, cols, &[matrix], &mut [&mut out]),
        4 => {
            for (xt, ot) in x.chunks_exact(cols).zip(out.chunks_exact_mut(rows)) {
                k.project_group_bf16(xt, cols, &[matrix], &mut [ot]);
            }
        }
        5 => k.project_group_bf16(x, cols, &[matrix], &mut [&mut out]),
        _ => panic!("mode"),
    }
    let bytes: Vec<u8> = out.iter().flat_map(|v| v.to_le_bytes()).collect();
    ziskos::io::commit_slice(&pci_f8_guests::sha(&bytes));
}
