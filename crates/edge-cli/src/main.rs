//! `edge` — the pipeline, running.
//!
//! Every crate in this workspace is a library, and until now nothing put them
//! together. That is not a cosmetic gap. The seams between the layers are where
//! the expensive bugs live, and a set of libraries each tested alone will
//! happily agree on nothing: the assembler's `MarketId` means one thing, the
//! strategy's another, and no test anywhere notices because no test ever holds
//! both at once.
//!
//! So this binary is deliberately thin and deliberately complete. It drives the
//! whole chain against the deterministic simulator:
//!
//! ```text
//! MarketSource → Assembler → MarketView → Strategy → RiskEngine → fills
//! ```
//!
//! What it is not is a trading system. There is no venue execution, no order
//! lifecycle, and no persistence; a filled order here is an assumption, not a
//! confirmation. It exists so the pipeline can be *run* and *watched*, and so
//! the integration tests have a shape to follow.
//!
//! ```text
//! cargo run --release -p edge-cli -- --ticks 400 --seed 7
//! ```

use std::collections::HashMap;

use edge_alpha::strategy::MarketView;
use edge_alpha::{QuoteConfig, QuoteMaker, Strategy};
use edge_core::fees::Liquidity;
use edge_core::types::{MarketId, Notional, Price, StrategyId, VenueId};
use edge_data::assembler::{Assembler, AssemblerConfig, Event};
use edge_data::source::MarketSource;
use edge_data::venues::sim::{SimConfig, Simulator};
use edge_risk::engine::RiskEngine;
use edge_risk::limits::{RiskDecision, RiskLimits};

const VENUE: VenueId = VenueId(1);
const MAKER: StrategyId = StrategyId(1);

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

const USAGE: &str = "\
edge — run the pipeline against the deterministic simulator

USAGE:
    edge [--ticks N] [--seed S] [--bankroll D]

OPTIONS:
    --ticks N       poll cycles to run (default 200)
    --seed S        simulator seed; the same seed replays identically
    --bankroll D    starting cash in dollars (default 10000)";

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
    let mut assembler = Assembler::new(VENUE, AssemblerConfig::default());
    let mut maker = QuoteMaker::new(MAKER, QuoteConfig::default());

    let bankroll = Notional::from_dollars(args.bankroll);
    let mut risk = RiskEngine::new(RiskLimits::for_bankroll(bankroll), bankroll);

    // The catalogue first: an update naming a ticker with no listing is dropped
    // rather than guessed at, so nothing at all happens without this.
    let listings = venue.listings().await.map_err(|e| e.to_string())?;
    for l in &listings {
        let (market, _) = assembler.register(l);
        if let Some(spec) = assembler.registry().get(market) {
            risk.portfolio_mut().set_event(market, spec.event_id);
        }
    }
    println!(
        "registered {} markets across {} listings",
        assembler.registry().len(),
        listings.len()
    );

    let mut tally = Tally::default();
    let mut events = Vec::new();
    let mut actions = Vec::new();

    for _ in 0..args.ticks {
        // Time is data: one clock, read here, threaded through everything below.
        let now = venue.now();

        events.clear();
        for update in venue.snapshot(&[], now).await.map_err(|e| e.to_string())? {
            assembler.apply(update, &mut events);
        }
        for e in &events {
            match e {
                Event::Gap { .. } => tally.gaps += 1,
                Event::Settled { market, outcome, .. } => {
                    risk.on_settle(*market, *outcome);
                    maker.on_settle(*market, *outcome);
                    tally.settled += 1;
                }
                _ => {}
            }
        }

        // Only fresh, non-stale markets are offered to the strategy. A stale
        // book is not a book to trade carefully — its best bid may be a price
        // nobody is offering.
        for market in assembler.fresh_markets(now) {
            let (Some(spec), Some(book)) =
                (assembler.registry().get(market), assembler.book(market))
            else {
                continue;
            };
            if !spec.status.is_tradable() {
                continue;
            }

            // Mark the market before asking anything of it. The risk engine
            // refuses to size an unmarked market on purpose — a stale or absent
            // price is the most common cause of a large accidental position —
            // so without this every order is rejected and the pipeline looks
            // like it is working while doing nothing.
            let Some(mid) = book.mid() else {
                continue; // one-sided book: no mid, nothing to mark or quote
            };
            risk.set_mark(market, mid, now);

            let position = risk.portfolio().qty(market);
            let avg_cost =
                risk.portfolio().position(market).map(|p| p.avg_cost().dollars()).unwrap_or(0.0);

            let view = MarketView {
                spec,
                book,
                // No model and no cross-venue consensus in this driver, so the
                // maker quotes around the market's own mid. Deliberate: a
                // predictor with no demonstrated skill must not move a price,
                // and pretending otherwise here would be the exact failure the
                // alpha crate is built to prevent.
                features: None,
                prediction: None,
                consensus: None,
                position,
                avg_cost,
                bankroll: risk.portfolio().cash.dollars(),
                resting: &[],
                now,
            };

            actions.clear();
            maker.on_market(&view, &mut actions);
            tally.intents += actions.len() as u64;

            for action in &actions {
                let Some(intent) = action.intent() else {
                    continue;
                };
                let fee = spec.fee.fee_per_contract(intent.price, intent.qty, Liquidity::Maker);
                let fee = Notional::from_dollars(fee);
                let decision = risk.check(market, intent.side, intent.price, intent.qty, fee, now);
                tally.record(&decision);

                let filled = decision.qty();
                if filled.get() > 0 {
                    // Assumed filled at the quoted price. This is the line where
                    // a real system would talk to a venue and wait to be told.
                    let total = Notional(fee.0 * filled.get());
                    risk.on_fill(market, intent.side, intent.price, filled, total, now);
                }
            }
        }

        risk.update(now);
        if let Some(reason) = risk.kill_reason() {
            println!("halted: {reason}");
            break;
        }
    }

    let marks = risk.marks();
    report(&args, &risk, &tally, &marks);
    Ok(())
}

#[derive(Debug, Default)]
struct Tally {
    intents: u64,
    approved: u64,
    resized: u64,
    rejected: u64,
    gaps: u64,
    settled: u64,
}

impl Tally {
    fn record(&mut self, d: &RiskDecision) {
        match d {
            RiskDecision::Approve(_) => self.approved += 1,
            RiskDecision::Resize(..) => self.resized += 1,
            RiskDecision::Reject(_) => self.rejected += 1,
        }
    }
}

fn report(args: &Args, risk: &RiskEngine, tally: &Tally, marks: &HashMap<MarketId, Price>) {
    let pf = risk.portfolio();
    println!();
    println!("-- session ----------------------------------------");
    println!("  seed                {}", args.seed);
    println!("  ticks               {}", args.ticks);
    println!("  gaps / settlements  {} / {}", tally.gaps, tally.settled);
    println!();
    println!("-- strategy ---------------------------------------");
    println!("  intents             {}", tally.intents);
    println!(
        "  approved/resized/rejected  {} / {} / {}",
        tally.approved, tally.resized, tally.rejected
    );
    println!();
    println!("-- book -------------------------------------------");
    println!("  cash                {:>12.2}", pf.cash.dollars());
    println!("  capital at risk     {:>12.2}", pf.capital_at_risk().dollars());
    println!("  realised            {:>12.2}", pf.realized().dollars());
    println!("  unrealised          {:>12.2}", pf.unrealized(marks).dollars());
    println!("  equity              {:>12.2}", pf.equity(marks).dollars());
    println!("  fees                {:>12.2}", pf.total_fees.dollars());
    println!("  open markets        {:>12}", pf.open_count());

    // The invariant worth printing every run: nothing in the ledger is
    // approximate, so cash plus what the positions cost has to equal the
    // starting bankroll plus everything realised, exactly.
    let reconciled = pf.cash + pf.capital_at_risk() - pf.realized();
    let ok = reconciled == pf.starting_cash;
    println!();
    println!("  reconciles          {}", if ok { "yes" } else { "NO — ledger bug" });
    if !ok {
        println!(
            "    {} vs {} (out by {})",
            reconciled.dollars(),
            pf.starting_cash.dollars(),
            (reconciled - pf.starting_cash).dollars()
        );
    }
}
