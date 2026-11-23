//! The `dot_f8` precompile behind the canonical kernel's backend trait.
use pci_dotf8::DotF8;
use ziskos::syscalls::{DOT_F8_E4M3, SyscallDotF8Params, dot_f8_formats, syscall_dot_f8};

pub struct Syscall;

impl DotF8 for Syscall {
    #[inline(always)]
    fn block_sums(&mut self, x: &[u64], y: &[u64], sums: &mut [i64]) {
        assert!(x.len() == sums.len() && y.len() == sums.len());
        let p = SyscallDotF8Params {
            x: x.as_ptr().cast(),
            y: y.as_ptr().cast(),
            n: 8 * sums.len() as u64,
            formats: dot_f8_formats(DOT_F8_E4M3, DOT_F8_E4M3),
            block: 8,
            out: sums.as_mut_ptr(),
        };
        // SAFETY: x and y are whole aligned words of n E4M3 values, sums holds n / 8
        // aligned i64 and is a distinct allocation from x, y and the parameter block.
        unsafe { syscall_dot_f8(&p) };
    }
}

pub fn sha(b: &[u8]) -> [u8; 32] {
    ziskos::zisklib::sha256(b)
}
