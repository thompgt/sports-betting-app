//! `edge` — the pipeline, running.
//!
//! A thin front end over [`edge_cli::Pipeline`], which is where the actual
//! wiring lives so that the integration tests can drive the same code with a
//! different venue behind it.
//!
//! What this is not is a trading system. There is no venue execution, no order
//! lifecycle, and no persistence; a filled order here is an assumption, not a
//! confirmation. It exists so the pipeline can be *run* and *watched* rather
//! than only unit-tested apart.
//!
//! ```text
//! cargo run --release -p edge-cli -- --ticks 400 --seed 7
//! ```

use edge_cli::{Pipeline, Tally};
use edge_core::types::{Notional, VenueId};
use edge_data::source::MarketSource;
use edge_data::venues::sim::{SimConfig, Simulator};
use edge_risk::engine::RiskEngine;

const VENUE: VenueId = VenueId(1);

/// Command-line options, parsed by hand.
///
/// No argument-parsing dependency: there are three flags, and a crate that
/// pulls in a dozen transitive dependencies to read them would be a poor trade
/// in a workspace whose whole argument is that its dependencies are chosen
/// deliberately.
#[derive(Debug, Clone, Copy)]
struct Args {
    ticks: usize,
    seed: u64,
    bankroll: f64,
}

impl Default for Args {
    fn default() -> Self {
        Args { ticks: 200, seed: 0xED9E, bankroll: 10_000.0 }
    }
}

const USAGE: &str = "\
edge — run the pipeline against the deterministic simulator

USAGE:
    edge [--ticks N] [--seed S] [--bankroll D]

OPTIONS:
    --ticks N       poll cycles to run (default 200)
    --seed S        simulator seed; the same seed replays identically
    --bankroll D    starting cash in dollars (default 10000)";

fn parse_args() -> Result<Args, String> {
    let mut args = Args::default();
    let mut argv = std::env::args().skip(1);
    while let Some(flag) = argv.next() {
        let mut value = || argv.next().ok_or(format!("{flag} needs a value"));
        match flag.as_str() {
            "--ticks" => args.ticks = value()?.parse().map_err(|e| format!("--ticks: {e}"))?,
            "--seed" => args.seed = value()?.parse().map_err(|e| format!("--seed: {e}"))?,
            "--bankroll" => {
                args.bankroll = value()?.parse().map_err(|e| format!("--bankroll: {e}"))?
            }
            "-h" | "--help" => {
                println!("{USAGE}");
                std::process::exit(0);
            }
            other => return Err(format!("unknown flag {other}\n\n{USAGE}")),
        }
    }
    Ok(args)
}

#[tokio::main]
async fn main() {
    let args = match parse_args() {
        Ok(a) => a,
        Err(e) => {
            eprintln!("edge: {e}");
            std::process::exit(2);
        }
    };
    if let Err(e) = run(args).await {
        eprintln!("edge: {e}");
        std::process::exit(1);
    }
}

async fn run(args: Args) -> Result<(), String> {
    let venue = Simulator::new(VENUE, SimConfig { seed: args.seed, ..SimConfig::default() });
    let bankroll = Notional::from_dollars(args.bankroll);
    let mut pipeline = Pipeline::new(VENUE, bankroll);

    let listings = venue.listings().await.map_err(|e| e.to_string())?;
    pipeline.register(&listings);
    println!(
        "registered {} markets across {} listings",
        pipeline.assembler.registry().len(),
        listings.len()
    );

    for _ in 0..args.ticks {
        // Time is data: one clock, read here, threaded through everything below.
        let now = venue.now();
        let updates = venue.snapshot(&[], now).await.map_err(|e| e.to_string())?;
        pipeline.tick(updates, now);

        if let Some(reason) = pipeline.risk.kill_reason() {
            println!("halted: {reason}");
            break;
        }
    }

    report(&args, &pipeline);
    Ok(())
}

fn report(args: &Args, pipeline: &Pipeline) {
    let risk: &RiskEngine = &pipeline.risk;
    let marks = risk.marks();
    let pf = risk.portfolio();
    let t: &Tally = &pipeline.tally;

    println!();
    println!("-- session ----------------------------------------");
    println!("  seed                {}", args.seed);
    println!("  ticks               {}", args.ticks);
    println!("  gaps / settlements  {} / {}", t.gaps, t.settled);
    println!();
    println!("-- strategy ---------------------------------------");
    println!("  intents             {}", t.intents);
    println!("  approved/resized/rejected  {} / {} / {}", t.approved, t.resized, t.rejected);
    println!("  orders sent         {}", pipeline.placed.len());
    println!();
    println!("-- book -------------------------------------------");
    println!("  cash                {:>12.2}", pf.cash.dollars());
    println!("  capital at risk     {:>12.2}", pf.capital_at_risk().dollars());
    println!("  realised            {:>12.2}", pf.realized().dollars());
    println!("  unrealised          {:>12.2}", pf.unrealized(&marks).dollars());
    println!("  equity              {:>12.2}", pf.equity(&marks).dollars());
    println!("  fees                {:>12.2}", pf.total_fees.dollars());
    println!("  open markets        {:>12}", pf.open_count());
    println!();
    println!("  reconciles          {}", if pipeline.reconciles() { "yes" } else { "NO — bug" });
}
