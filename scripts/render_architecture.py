"""
Render docs/assets/architecture.png -- the LineEdge data-flow diagram.

Uses matplotlib rather than graphviz so the diagram builds with nothing but the
Python requirements (no system `dot` binary needed).

Usage:
    python scripts/render_architecture.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path(__file__).resolve().parents[1] / "docs" / "assets" / "architecture.png"

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
MUTED = "#898781"
BORDER = "#c3c2b7"

STAGE = "#2a78d6"   # categorical slot 1 - the poll/decide pipeline
STORE = "#eb6834"   # slot 2 - persistence
SINK = "#1baf7a"    # slot 3 - consumers

FONT = {"family": "DejaVu Sans"}

W, H = 13.9, 8.9
BOX_W, BOX_H, GAP = 2.9, 1.45, 0.42
ROW1_Y = 5.75


def box(ax, x, y, w, h, accent, title, lines, title_size=11.5, body_size=9.0):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.14",
        linewidth=1.6, edgecolor=accent, facecolor="white", zorder=2,
    ))
    # accent rule along the top edge
    ax.plot([x + 0.14, x + w - 0.14], [y + h - 0.055, y + h - 0.055],
            color=accent, linewidth=3.2, solid_capstyle="round", zorder=3)

    ax.text(x + w / 2, y + h - 0.42, title, ha="center", va="center",
            fontsize=title_size, color=INK, fontweight="bold", zorder=4, **FONT)
    for i, line in enumerate(lines):
        ax.text(x + w / 2, y + h - 0.78 - i * 0.29, line, ha="center", va="center",
                fontsize=body_size, color=INK_SECONDARY, zorder=4, **FONT)


def arrow(ax, start, end, color=BORDER, style="-|>", rad=0.0, lw=1.8, dashed=False):
    ax.add_patch(FancyArrowPatch(
        start, end,
        arrowstyle=style, mutation_scale=16,
        connectionstyle=f"arc3,rad={rad}",
        linewidth=lw, color=color, zorder=1,
        linestyle="--" if dashed else "-",
    ))


def label(ax, x, y, text, size=8.5, color=MUTED, ha="center", style="italic"):
    ax.text(x, y, text, ha=ha, va="center", fontsize=size, color=color,
            fontstyle=style, zorder=5, **FONT)


def main() -> None:
    fig, ax = plt.subplots(figsize=(W, H), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")

    xs = [0.45 + i * (BOX_W + GAP) for i in range(4)]

    ax.text(0.45, H - 0.35, "LineEdge — poll / price / detect / persist",
            fontsize=15, color=INK, fontweight="bold", ha="left", va="center", **FONT)
    ax.text(0.45, H - 0.75,
            "EdgeDetectionService.run_forever() repeats the top row on an interval; nothing places bets.",
            fontsize=9.5, color=INK_SECONDARY, ha="left", va="center", **FONT)

    box(ax, xs[0], ROW1_Y, BOX_W, BOX_H, STAGE, "1 · Ingest",
        ["OddsProvider.get_odds()", "MockOddsClient today;", "any API can implement it"])
    box(ax, xs[1], ROW1_Y, BOX_W, BOX_H, STAGE, "2 · Resolve",
        ["EntityResolver", "exact map → rapidfuzz", "+ 6h start-time window"])
    box(ax, xs[2], ROW1_Y, BOX_W, BOX_H, STAGE, "3 · Price",
        ["engine/math_utils.py", "consensus implied probs", "→ power-method devig"])
    box(ax, xs[3], ROW1_Y, BOX_W, BOX_H, STAGE, "4 · Detect",
        ["EV vs fair price", "threshold + EdgeCache", "dedupe (TTL / EV spike)"])

    for i in range(3):
        arrow(ax, (xs[i] + BOX_W, ROW1_Y + BOX_H / 2), (xs[i + 1] - 0.06, ROW1_Y + BOX_H / 2))

    # storage
    store_x, store_y, store_w, store_h = xs[1], 3.5, BOX_W * 3 + GAP * 2, 1.25
    box(ax, store_x, store_y, store_w, store_h, STORE, "SQLite  ·  storage/",
        ["detected_edges (odds_offered, fair_odds, calculated_ev, outcome_name, closing_line, clv_pct)",
         "canonical_teams / canonical_games — seeded once, loaded into the resolver"],
        body_size=8.8)

    # detect -> storage
    arrow(ax, (xs[3] + BOX_W / 2, ROW1_Y - 0.06), (xs[3] + BOX_W / 2, store_y + store_h + 0.06),
          color=STORE)
    label(ax, xs[3] + BOX_W / 2 + 1.35, (ROW1_Y + store_y + store_h) / 2, "write +EV edge")

    # storage -> resolver (canonical entities)
    arrow(ax, (xs[1] + 0.55, store_y + store_h + 0.06), (xs[1] + 0.55, ROW1_Y - 0.06),
          color=BORDER, dashed=True)
    label(ax, xs[1] - 0.62, (ROW1_Y + store_y + store_h) / 2, "canonical\nentities", ha="center")

    # consumers
    aud_x, sink_y, sink_h = xs[1], 0.9, 1.4
    aud_w = BOX_W * 1.5 + GAP / 2
    dash_x = aud_x + aud_w + GAP
    dash_w = store_x + store_w - dash_x

    box(ax, aud_x, sink_y, aud_w, sink_h, SINK, "EdgeAuditor · CLV",
        ["once a game starts: stamp closing_line,", "compute clv_pct, deactivate the edge"],
        body_size=8.8)
    box(ax, dash_x, sink_y, dash_w, sink_h, SINK, "Streamlit dashboard",
        ["live +EV table, detection history,", "CLV audit — reads the same DB"],
        body_size=8.8)

    arrow(ax, (aud_x + aud_w * 0.35, store_y - 0.06), (aud_x + aud_w * 0.35, sink_y + sink_h + 0.06),
          color=SINK)
    arrow(ax, (aud_x + aud_w * 0.72, sink_y + sink_h + 0.06), (aud_x + aud_w * 0.72, store_y - 0.06),
          color=SINK)
    label(ax, aud_x + aud_w * 0.53, (store_y + sink_y + sink_h) / 2 + 0.02, "read /\nwrite back")

    arrow(ax, (dash_x + dash_w * 0.5, store_y - 0.06), (dash_x + dash_w * 0.5, sink_y + sink_h + 0.06),
          color=SINK)
    label(ax, dash_x + dash_w * 0.5 + 0.72, (store_y + sink_y + sink_h) / 2 + 0.02, "read")

    # poll loop feedback: stage 4 wraps back around to stage 1
    top = ROW1_Y + BOX_H
    loop_y = top + 0.5
    x_from, x_to = xs[3] + BOX_W - 0.5, xs[0] + 0.5
    arrow(ax, (x_from, top + 0.04), (x_from, loop_y), color=STAGE, style="-", lw=1.6)
    arrow(ax, (x_from, loop_y), (x_to, loop_y), color=STAGE, style="-", lw=1.6)
    arrow(ax, (x_to, loop_y), (x_to, top + 0.04), color=STAGE, lw=1.6)
    label(ax, (x_from + x_to) / 2, loop_y + 0.24,
          "poll_interval_seconds  ·  exponential backoff on failure", color=STAGE, size=9)

    fig.savefig(OUT, facecolor=SURFACE, bbox_inches="tight", pad_inches=0.28)
    print(f"Wrote {OUT} ({OUT.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
