//! Recursion probe: set up the PCI join recurser (`../recursion`) on the pinned
//! recursive proving key and fold two real child proofs once.
//!
//! pci-aggregate-host EMBED.elf LAYER.elf HEAD.elf PROOF_A PROOF_B OUT_DIR
//!
//! Environment: ZISK_HOME must hold `provingKey` (a link to the pinned
//! setup-rec key); the recurser is written under `$ZISK_HOME/recurser`.
//! Writes OUT_DIR/aggregate.proof.bin, OUT_DIR/recurser.verkey.json and
//! OUT_DIR/aggregate.json (publics), with wall times on stderr.
use std::error::Error;
use std::time::Instant;
use zisk_sdk::{AggregationProgramBuilder, CircomCircuit, GuestProgram, Proof, ProverClient};

const RECURSION: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../recursion/circuits");

#[tokio::main(flavor = "multi_thread")]
async fn main() -> Result<(), Box<dyn Error>> {
    let a: Vec<String> = std::env::args().collect();
    if a.len() != 7 {
        return Err("usage: pci-aggregate-host EMBED.elf LAYER.elf HEAD.elf PROOF_A PROOF_B OUT_DIR".into());
    }
    let out = std::path::Path::new(&a[6]);
    std::fs::create_dir_all(out)?;
    let embed = GuestProgram::from_uri(&a[1])?;
    let layer = GuestProgram::from_uri(&a[2])?;
    let head = GuestProgram::from_uri(&a[3])?;
    let recurser = AggregationProgramBuilder::new(
        CircomCircuit::from_path(format!("{RECURSION}/aggregate.circom"))?,
        5,
    )
    .programs(&[&embed, &layer, &head])
    .normalize(CircomCircuit::from_path(format!("{RECURSION}/normalize.circom"))?)
    .build()?;
    eprintln!("recurser_id {}", recurser.recurser_id());

    let client = ProverClient::embedded().gpu().build()?;
    let t = Instant::now();
    client.setup(&recurser).run()?.await?;
    let vk = recurser.vk()?;
    eprintln!("recurser setup {:.1} s, vk {:?}", t.elapsed().as_secs_f64(), vk.vk);
    std::fs::write(out.join("recurser.verkey.json"), format!("{:?}\n", vk.vk))?;

    let pa = Proof::load(&a[4])?;
    let pb = Proof::load(&a[5])?;
    let t = Instant::now();
    let agg = client.aggregate_proofs(&recurser, &pa, &pb).run()?.await?.get_proof().clone();
    eprintln!("fold {:.1} s", t.elapsed().as_secs_f64());
    agg.save(out.join("aggregate.proof.bin"))?;
    let publics = agg.publics_full().map(|p| p.to_vec()).unwrap_or_default();
    std::fs::write(out.join("aggregate.json"), format!("{{\"publics_full\": {publics:?}}}\n"))?;
    eprintln!("aggregate publics_full {publics:?}");
    Ok(())
}
