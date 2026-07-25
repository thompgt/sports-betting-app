"""
Capture real screenshots of the Streamlit dashboard for the README.

Launches the dashboard headlessly against a demo database (produced by
`scripts/seed_demo_db.py`, which contains SIMULATED market data), walks each
tab with Playwright, and writes PNGs to docs/assets/.

Every capture is validated before it is kept: the page must not be showing a
Streamlit exception, and the PNG must be large enough to be actual content
rather than a blank frame.

Usage:
    $env:PYTHONPATH = "."; python scripts/seed_demo_db.py
    $env:PYTHONPATH = "."; python scripts/capture_docs.py
"""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
from pathlib import Path

from playwright.async_api import async_playwright

REPO_ROOT = Path(__file__).resolve().parents[1]
ASSETS = REPO_ROOT / "docs" / "assets"
DEMO_DB = REPO_ROOT / "lineedge_demo.db"
PORT = 8501
MIN_PNG_BYTES = 20_000

# (tab index, output filename)
TABS = [
    (0, "dashboard_live_edges.png"),
    (1, "dashboard_historical.png"),
    (2, "dashboard_clv_audit.png"),
]
# The tab used as the README hero image.
HERO_TAB_INDEX = 0
HERO_FILENAME = "dashboard_preview.png"


def start_server() -> subprocess.Popen:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT)
    env["LINEEDGE_DB_URL"] = f"sqlite:///{DEMO_DB.as_posix()}"
    return subprocess.Popen(
        [
            sys.executable, "-m", "streamlit", "run", "app/ui/dashboard.py",
            "--server.port", str(PORT),
            "--server.headless", "true",
            "--browser.gatherUsageStats", "false",
            "--client.toolbarMode", "minimal",
        ],
        cwd=str(REPO_ROOT),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )


async def wait_for_render(page) -> None:
    """Give Streamlit's websocket rerun + Plotly a chance to settle."""
    await page.wait_for_selector('[data-testid="stAppViewContainer"]', timeout=60_000)
    try:
        await page.wait_for_selector('[data-testid="stStatusWidget"]', state="detached", timeout=15_000)
    except Exception:
        pass  # the "Running" widget may never appear on a fast rerun
    await asyncio.sleep(3.0)


async def assert_healthy(page, label: str) -> None:
    if await page.locator('[data-testid="stException"]').count():
        text = await page.locator('[data-testid="stException"]').first.inner_text()
        raise RuntimeError(f"Dashboard raised an exception on {label}:\n{text}")
    body = await page.locator("body").inner_text()
    if "Traceback (most recent call last)" in body:
        raise RuntimeError(f"Dashboard rendered a traceback on {label}")
    if "No data found in the database" in body:
        raise RuntimeError(
            f"Dashboard has no data on {label}. Run scripts/seed_demo_db.py first."
        )


def validate_png(path: Path, label: str) -> int:
    size = path.stat().st_size if path.exists() else 0
    if size < MIN_PNG_BYTES:
        raise RuntimeError(f"{label}: {path.name} is only {size} bytes - likely blank.")
    return size


async def capture() -> None:
    if not DEMO_DB.exists():
        raise SystemExit(
            f"{DEMO_DB} not found. Run: $env:PYTHONPATH='.'; python scripts/seed_demo_db.py"
        )

    ASSETS.mkdir(parents=True, exist_ok=True)
    print("Launching Streamlit...")
    process = start_server()
    results = []
    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch()
            # Streamlit's main block scrolls inside its own container, so Playwright's
            # full_page capture cannot grow past the viewport -- the viewport itself has
            # to be tall enough to hold the whole tab.
            context = await browser.new_context(
                viewport={"width": 1440, "height": 1500}, device_scale_factor=2
            )
            page = await context.new_page()

            # Streamlit's HTTP port opens before the app is servable; navigating
            # too early yields a blank document that never hydrates.
            print("Waiting for Streamlit to become servable...")
            await asyncio.sleep(12.0)

            print(f"Navigating to http://localhost:{PORT} ...")
            for attempt in range(10):
                try:
                    await page.goto(
                        f"http://localhost:{PORT}", timeout=30_000, wait_until="networkidle"
                    )
                    await wait_for_render(page)
                    if await page.locator('button[data-baseweb="tab"]').count():
                        break
                except Exception:
                    if attempt == 9:
                        raise
                await asyncio.sleep(3.0)
            else:
                raise RuntimeError("Dashboard never rendered its tabs.")

            tab_buttons = page.locator('button[data-baseweb="tab"]')
            count = await tab_buttons.count()
            print(f"Found {count} dashboard tabs.")

            for index, filename in TABS:
                if index >= count:
                    raise RuntimeError(f"Tab index {index} not present (only {count} tabs).")
                await tab_buttons.nth(index).click()
                await asyncio.sleep(3.5)  # Plotly needs a beat to draw
                await assert_healthy(page, filename)

                out = ASSETS / filename
                await page.screenshot(path=str(out), full_page=True)
                size = validate_png(out, filename)
                results.append((filename, size))
                print(f"  captured {filename} ({size:,} bytes)")

                if index == HERO_TAB_INDEX:
                    hero = ASSETS / HERO_FILENAME
                    await page.screenshot(path=str(hero), full_page=True)
                    results.append((HERO_FILENAME, validate_png(hero, HERO_FILENAME)))
                    print(f"  captured {HERO_FILENAME}")

            await browser.close()
    finally:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()

    print(f"\n{len(results)} screenshots written to {ASSETS}")


if __name__ == "__main__":
    asyncio.run(capture())
