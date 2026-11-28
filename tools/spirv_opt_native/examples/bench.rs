//! `cargo run --release --example bench -- <dir>`: times `optimize()` over every file in a folder of .spvasm.
//!
//! Modules the port declines (they stay in Python) are counted and left out of the throughput: MB/s is over the
//! accepted bytes only, and the run fails when none is accepted.
fn main() {
    let dir = std::env::args().nth(1).expect("usage: bench <dir of .spvasm files>");
    let mut mods = Vec::new();
    for e in std::fs::read_dir(dir).expect("read_dir") {
        mods.push(std::fs::read(e.expect("entry").path()).expect("read"));
    }
    // Declined modules are reported apart and left out of the timing.
    let accepted: Vec<&Vec<u8>> = mods.iter().filter(|m| spirv_opt_native::optimize(m, 4).is_ok()).collect();
    let declined = mods.len() - accepted.len();
    let accepted_bytes: usize = accepted.iter().map(|m| m.len()).sum();
    let declined_bytes: usize = mods.iter().map(|m| m.len()).sum::<usize>() - accepted_bytes;
    if accepted.is_empty() {
        eprintln!("no module accepted: {} declined ({:.2} MB), nothing to time", declined, declined_bytes as f64 / 1e6);
        std::process::exit(1);
    }
    let mut best = f64::MAX;
    for _ in 0..7 {
        let t = std::time::Instant::now();
        for m in &accepted {
            std::hint::black_box(spirv_opt_native::optimize(m, 4).is_ok());
        }
        best = best.min(t.elapsed().as_secs_f64());
    }
    println!("{} modules accepted, {:.2} MB: {:.1} ms ({:.0} MB/s); {} declined, {:.2} MB, not timed",
             accepted.len(), accepted_bytes as f64 / 1e6, best * 1e3, accepted_bytes as f64 / 1e6 / best, declined,
             declined_bytes as f64 / 1e6);
}
