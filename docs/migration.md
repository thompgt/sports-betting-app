# Migration: LineEdge (Python) → Edge (Rust)

## Why

LineEdge detects mispricings and writes them to a database. That is where a
detector stops and a trading system starts. The rebuild targets the parts that
were missing, and they are the parts that decide whether the thing makes money:

| LineEdge (Python) | Edge (Rust) |
|---|---|
| Detection only, no execution | Order management, matching, paper and live execution |
| Gross EV | EV net of venue fees, which usually exceed the edge |
| Sportsbook odds feeds | Prediction-market venues (Kalshi, Polymarket) plus sportsbooks |
| Mean of book probabilities | Log-odds pooling with dispersion feeding position sizing |
| Unsafeguarded Newton devig | Bracketed solvers that cannot diverge |
| Kelly computed, never used | Kelly sized, risk-checked, and actually routed |
| No risk layer | Pre-trade limits, VaR/CVaR, drawdown kill switch |
| `datetime.utcnow()` throughout | Time is data, so backtest and live share one code path |
| Streamlit read-only dashboard | HTTP + WebSocket API with a live dashboard |

Rust rather than C++ or Java: the latency profile of C++ without the memory
safety risk in a system that moves money, and a mature async/HTTP/SQLite
ecosystem so the production half is achievable rather than aspirational.

## Architecture

```
crates/
  edge-core     Pure quant: types, odds, devig, consensus, fees, EV/Kelly, stats
  edge-book     Limit order book, matching engine, LMSR and CPMM market makers
  edge-risk     Pre-trade limits, VaR/CVaR, drawdown control, kill switch
  edge-alpha    Feature extraction, online predictor, strategies
  edge-data     Venue adapters, entity resolution, SQLite store, event journal
  edge-engine   Runtime wiring, execution simulator, backtester
  edge-server   HTTP + WebSocket API and dashboard
  edge-cli      Operator command line
```

`edge-core` is pure — no I/O, no clock, no globals. Everything above it is a
function of its input event stream, which is what makes a backtest evidence
about the code that will actually run.

## Status

| # | Component | State |
|---|---|---|
| 1 | `edge-core` — types, odds, devig, consensus, fees, EV/Kelly, stats | ✅ done, 103 tests |
| 2 | `edge-book` — order book and matching engine | ✅ done, 88 tests |
| 3 | `edge-book` — LMSR and CPMM | ✅ done |
| 4 | `edge-risk` — limits, VaR, sizing, drawdown | ✅ done, 68 tests |
| 5 | `edge-alpha` — strategy framework and strategies | ✅ done, 106 tests |
| 6 | `edge-alpha` — online price predictor | ✅ done |
| 7 | `edge-data` — ingestion and venue adapters | ⏳ |
| 8 | `edge-data` — persistence and event journal | ⏳ |
| 9 | `edge-engine` — runtime | ⏳ |
| 10 | `edge-engine` — backtester | ⏳ |
| 11 | `edge-server` — API and dashboard | ⏳ |
| 12 | CLI, config, CI, docs | ⏳ |

## Building on Windows

Rust's `windows-gnu` target links against msvcrt. MSYS2's default UCRT toolchain
does not match it, and proc-macro DLLs fail to link with a bare `ld` error 116.
Install the msvcrt-based compiler and put it first on `PATH`:

```powershell
C:\msys64\usr\bin\pacman -S --needed mingw-w64-x86_64-gcc
$env:PATH = "C:\msys64\mingw64\bin;$env:PATH"
cargo test
```

The `windows-msvc` target works without any of this if Visual Studio Build Tools
are installed, and is the better choice where they are available.
