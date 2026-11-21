//! Runs one Qwen3 segment and commits sha256(output), and with `HASH_INPUT`
//! also sha256(input).

#![no_main]

ziskos::entrypoint!(main);

use qwen_common::{HASH_INPUT, Header, run};
use ziskos::zisklib::sha256;

fn main() {
    let input = ziskos::io::read_slice();
    let h = Header::parse(&input);
    #[cfg(feature = "floatdot")]
    let dot: qwen_common::DotFn = if h.flags & qwen_common::FLOATDOT != 0 {
        qwen_common::dot_floatdot
    } else {
        qwen_common::dot_soft
    };
    #[cfg(not(feature = "floatdot"))]
    let dot: qwen_common::DotFn = qwen_common::dot_soft;
    #[cfg(feature = "dotf8")]
    let qdot: qwen_common::QDotFn = qwen_common::qdot_dotf8;
    #[cfg(not(feature = "dotf8"))]
    let qdot: qwen_common::QDotFn = qwen_common::qdot_soft;
    let out = run(&input, dot, qdot);
    let bytes: Vec<u8> = out.iter().flat_map(|v| v.to_le_bytes()).collect();
    let mut output = sha256(&bytes).to_vec();
    if h.flags & HASH_INPUT != 0 {
        output.extend_from_slice(&sha256(&input));
    }
    ziskos::io::commit_slice(&output);
}
