// Native replay from little-endian NPY payloads and the original checkpoint range.
use pci_dotf8::{Kernel, SoftDotF8};
use std::{
    env,
    fs::File,
    io::{Read, Seek, SeekFrom},
};

fn read(path: &str, offset: &str, len: usize) -> Vec<u8> {
    let mut file = File::open(path).unwrap();
    file.seek(SeekFrom::Start(offset.parse().unwrap())).unwrap();
    let mut buf = vec![0; len];
    file.read_exact(&mut buf).unwrap();
    buf
}

fn main() {
    let args: Vec<_> = env::args().skip(1).collect();
    let [x, x_offset, w, w_offset, y, y_offset, positions, rows, cols] = args.as_slice() else {
        panic!("expected x offset weights offset out offset positions rows cols")
    };
    let t = positions.parse::<usize>().unwrap();
    let r = rows.parse::<usize>().unwrap();
    let k = cols.parse::<usize>().unwrap();
    let bits: Vec<_> = read(x, x_offset, t * k * 4)
        .chunks_exact(4)
        .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
        .collect();
    let weights = read(w, w_offset, r * k * 2);
    let expected = read(y, y_offset, t * r * 4);
    let mut actual = vec![0.0; t * r];
    Kernel::new(SoftDotF8).project_group_bf16(&bits, k, &[&weights], &mut [&mut actual]);
    for (i, (value, bytes)) in actual.iter().zip(expected.chunks_exact(4)).enumerate() {
        assert_eq!(
            value.to_le_bytes().as_slice(),
            bytes,
            "{y}: position {}, row {}",
            i / r,
            i % r
        );
    }
}
