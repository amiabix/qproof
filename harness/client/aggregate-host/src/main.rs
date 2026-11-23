//! Fold a pair of proofs or the complete, ordered Qwen proof roster.
//!
//! pci-aggregate-host EMBED.elf LAYER.elf HEAD.elf PROOF_A PROOF_B OUT_DIR
//!
//! pci-aggregate-host EMBED.elf LAYER.elf HEAD.elf --work WORK OUT_DIR
//!
//! Environment: ZISK_HOME must hold `provingKey` (a link to the pinned
//! setup-rec key); the recurser is written under `$ZISK_HOME/recurser`.
//! Writes OUT_DIR/aggregate.proof.bin, OUT_DIR/recurser.verkey.json and
//! OUT_DIR/aggregate.json (publics), with wall times on stderr.
mod tree;

use std::error::Error;
use std::path::{Path, PathBuf};
use std::time::Instant;
use zisk_sdk::{AggregationProgramBuilder, CircomCircuit, GuestProgram, Proof, ProverClient};

const RECURSION: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../recursion/circuits");

fn roster(work: &Path) -> Result<Vec<PathBuf>, Box<dyn Error>> {
    let text = std::fs::read_to_string(work.join("roster"))?;
    let names: Vec<_> = text.lines().collect();
    let heads = names.len().checked_sub(37).ok_or("incomplete roster")?;
    if !(4..=18).any(|log2| 151_936usize.div_ceil(1 << log2) == heads) {
        return Err("unsupported number of Qwen head shards".into());
    }
    let expected = std::iter::once("embed".to_owned())
        .chain((0..36).map(|i| format!("layer{i:02}")))
        .chain((0..heads).map(|i| format!("head-{i:05}")));
    if !expected.eq(names.iter().copied()) {
        return Err("unexpected child order".into());
    }
    let paths: Vec<_> = names
        .iter()
        .map(|n| work.join(format!("proofs/{n}.proof.bin")))
        .collect();
    if paths.iter().any(|p| !p.is_file()) {
        return Err("missing child proof".into());
    }
    Ok(paths)
}

#[tokio::main(flavor = "multi_thread")]
async fn main() -> Result<(), Box<dyn Error>> {
    let a: Vec<String> = std::env::args().collect();
    if a.len() != 7 {
        return Err("usage: pci-aggregate-host EMBED.elf LAYER.elf HEAD.elf (PROOF_A PROOF_B | --work WORK) OUT_DIR".into());
    }
    let paths = if a[4] == "--work" {
        roster(Path::new(&a[5]))?
    } else {
        vec![PathBuf::from(&a[4]), PathBuf::from(&a[5])]
    };
    let out = std::path::Path::new(&a[6]);
    std::fs::create_dir_all(out)?;
    let embed = GuestProgram::from_uri(&a[1])?;
    let layer = GuestProgram::from_uri(&a[2])?;
    let head = GuestProgram::from_uri(&a[3])?;
    let circuits = std::env::var_os("PCI_RECURSION_CIRCUITS")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(RECURSION));
    let recurser = AggregationProgramBuilder::new(
        CircomCircuit::from_path(circuits.join("aggregate.circom"))?,
        5,
    )
    .programs(&[&embed, &layer, &head])
    .normalize(CircomCircuit::from_path(circuits.join("normalize.circom"))?)
    .build()?;
    eprintln!("recurser_id {}", recurser.recurser_id());

    let client = ProverClient::embedded().gpu().build()?;
    let t = Instant::now();
    client.setup(&recurser).run()?.await?;
    let vk = recurser.vk()?;
    eprintln!(
        "recurser setup {:.1} s, vk {:?}",
        t.elapsed().as_secs_f64(),
        vk.vk
    );
    std::fs::write(out.join("recurser.verkey.json"), format!("{:?}\n", vk.vk))?;

    let t = Instant::now();
    let mut steps = Vec::with_capacity(2 * paths.len() - 1);
    tree::plan(0, paths.len(), &mut steps);
    let mut stack = Vec::new();
    for step in steps {
        match step {
            tree::Step::Leaf(i) => stack.push(Proof::load(&paths[i])?),
            tree::Step::Join => {
                let right = stack.pop().ok_or("missing right child")?;
                let left = stack.pop().ok_or("missing left child")?;
                stack.push(
                    client
                        .aggregate_proofs(&recurser, &left, &right)
                        .run()?
                        .await?
                        .get_proof()
                        .clone(),
                );
            }
        }
    }
    let agg = stack.pop().ok_or("empty aggregate")?;
    eprintln!("fold {:.1} s", t.elapsed().as_secs_f64());
    agg.save(out.join("aggregate.proof.bin"))?;
    let publics = agg.publics_full().map(|p| p.to_vec()).unwrap_or_default();
    std::fs::write(
        out.join("aggregate.json"),
        format!("{{\"publics_full\": {publics:?}}}\n"),
    )?;
    eprintln!("aggregate publics_full {publics:?}");
    Ok(())
}
