//! pci-layer-bench check  <layer.v1.stdin> [expected digest hex] [output dir]
//! pci-layer-bench sample <layer.v1.stdin> <output dir> <rows per projection> <positions,...>
use pci_layer::{digest_of, execute};
use pci_layer_bench::*;
use std::time::Instant;

type Res<T> = Result<T, Box<dyn std::error::Error>>;

fn timed<T>(name: &str, f: impl FnOnce() -> T) -> T {
    let s = Instant::now();
    let r = f();
    eprintln!("{name}: {:.1} s", s.elapsed().as_secs_f64());
    r
}

fn check(args: &[String]) -> Res<()> {
    let stdin = std::fs::read(&args[0])?;
    let record = record_of_stdin(&stdin);
    let reference = timed("reference (execute, Mac BF16 reference per position)", || execute(record, sha))?;
    let digest = digest_of(record, &reference.receipt);
    let bits = |o: &[f32]| -> Vec<u8> { o.iter().flat_map(|v| v.to_le_bytes()).collect() };
    let out = bits(&reference.output);
    println!("positions {}", reference.output.len() / 2560);
    println!("digest {}", hex(&digest));
    println!("output sha256 {}", hex(&sha(&out)));
    for (name, run) in [
        ("batched (execute_batched, canonical kernel, soft dot_f8)", batched as fn(&[u8]) -> _),
        ("per-position (execute_with, canonical kernel, soft dot_f8)", per_position),
    ] {
        let o = timed(name, || run(record))?;
        let same = o.receipt == reference.receipt
            && digest_of(record, &o.receipt) == digest
            && bits(&o.output) == out;
        println!("{name}: receipt, digest and output bits equal to reference: {same}");
        if !same {
            return Err(format!("{name} differs from the reference").into());
        }
    }
    if let Some(e) = args.get(1) {
        let ok = hex(&digest) == *e;
        println!("digest equals expected {e}: {ok}");
        if !ok {
            return Err("digest differs from expected".into());
        }
    }
    if let Some(d) = args.get(2) {
        std::fs::create_dir_all(d)?;
        std::fs::write(format!("{d}/output.f32"), &out)?;
        std::fs::write(format!("{d}/digest"), digest)?;
    }
    Ok(())
}

fn sample(args: &[String]) -> Res<()> {
    let stdin = std::fs::read(&args[0])?;
    let record = record_of_stdin(&stdin);
    let dir = std::path::PathBuf::from(&args[1]);
    let want_rows: usize = args[2].parse()?;
    let counts: Vec<usize> = args[3].split(',').map(|s| s.parse()).collect::<Result<_, _>>()?;
    std::fs::create_dir_all(&dir)?;
    // Capture each projection's real input (all positions) and matrix from the layer.
    let mut groups: Vec<(Vec<u32>, usize, Vec<Vec<u8>>)> = vec![];
    let o = pci_layer::execute_batched(record, sha, &mut |x, cols, m, outs| {
        groups.push((x.to_vec(), cols, m.iter().map(|b| b.to_vec()).collect()));
        pci_layer::project_batch_bf16(x, cols, m, outs);
    })?;
    let names = ["q", "k", "v", "o", "gate", "up", "down"];
    let mut manifest = String::new();
    let mut m = 0;
    for (x, cols, mats) in &groups {
        let total = x.len() / cols;
        for mat in mats {
            let rows = mat.len() / (2 * cols);
            // Evenly spaced real rows.
            let stride = rows / want_rows.min(rows);
            let picked: Vec<u8> = (0..want_rows.min(rows))
                .flat_map(|i| mat[2 * cols * i * stride..][..2 * cols].iter().copied())
                .collect();
            for &n in counts.iter().filter(|&&n| n <= total) {
                let xs = &x[..n * cols];
                let k = picked.len() / (2 * cols);
                let mut want = vec![0f32; n * k];
                pci_layer::project_batch_bf16(xs, *cols, &[&picked], &mut [&mut want]);
                let digest = sha(&want.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>());
                for mode in 3..=5 {
                    let name = format!("{}.t{n}.m{mode}.stdin", names[m]);
                    std::fs::write(dir.join(&name), stdin_of(&sample_payload(mode, n, *cols, xs, &picked)))?;
                }
                manifest += &format!("{} {n} {k} {cols} {}\n", names[m], hex(&digest));
            }
            m += 1;
        }
    }
    std::fs::write(dir.join("manifest.txt"), &manifest)?;
    println!("layer digest {}; samples in {}", hex(&digest_of(record, &o.receipt)), dir.display());
    print!("{manifest}");
    Ok(())
}

fn main() -> Res<()> {
    let a: Vec<String> = std::env::args().skip(1).collect();
    match a.first().map(String::as_str) {
        Some("check") if a.len() >= 2 => check(&a[1..]),
        Some("sample") if a.len() == 5 => sample(&a[1..]),
        _ => Err("usage: check <stdin> [digest] [dir] | sample <stdin> <dir> <rows> <n,n,..>".into()),
    }
}
