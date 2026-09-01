#!/usr/bin/env python3
"""Functional browser/WASM end-to-end tests for Zchezz.

The suite validates behavior, not machine speed. It waits for observable
conditions (engine ready, PV returned, move returned) instead of sleeping for
fixed durations shorter than the application's own search budget.

Covered contracts:
  B1  WASM worker initializes and exposes window.zchezzSearch.
  B2  Clock widgets are visible in Game mode and hidden in Depth mode.
  B3  Analysis returns a score and a non-empty PV.
  B4  MultiPV returns a real second line.
  B5  Clock-mode search returns a move when the opening book is unavailable.

Usage:
  python tests/test_browser.py --version v403 --html zchezz_bundle.html --headless
"""
from __future__ import annotations

import argparse
import functools
import http.server
import io
import os
import socketserver
import sys
import threading
from pathlib import Path

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]

# ═══════════════ CONFIGURATION ═══════════════
VERSION = ""
HTML_FILE = "zchezz_bundle.html"
PORT = 8766
HEADLESS = False
VIEWPORT_W = 1400
VIEWPORT_H = 950
ENGINE_READY_TIMEOUT_MS = 30_000
SEARCH_TIMEOUT_MS = 30_000
ANALYSIS_DEPTH = "5"
SCREENSHOT_DIR = ROOT / "artifacts" / "browser"
# ═════════════════════════════════════════════

passed = 0
failed = 0
errors_list: list[str] = []


class ReusableTCPServer(socketserver.TCPServer):
    allow_reuse_address = True


def check(name: str, condition: bool, detail: str = "") -> None:
    global passed, failed
    if condition:
        passed += 1
        print(f"PASS: {name}")
    else:
        failed += 1
        msg = f"FAIL: {name}" + (f" -- {detail}" if detail else "")
        print(msg)
        errors_list.append(msg)


def engine_dir(version: str) -> Path:
    if version:
        path = ROOT / "engine" / "c" / f"zchezz_{version}"
        if path.is_dir():
            return path
        raise FileNotFoundError(f"engine directory not found: {path}")

    versions = []
    for path in (ROOT / "engine" / "c").glob("zchezz_v*"):
        suffix = path.name.removeprefix("zchezz_v")
        if suffix.isdigit():
            versions.append((int(suffix), path))
    if not versions:
        raise FileNotFoundError("no numeric engine/c/zchezz_v* directory found")
    return max(versions)[1]


def start_server(directory: Path, port: int):
    handler = functools.partial(
        http.server.SimpleHTTPRequestHandler,
        directory=str(directory),
    )
    httpd = ReusableTCPServer(("127.0.0.1", port), handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    return httpd


def debug_log(page) -> str:
    try:
        locator = page.locator("#debug-log")
        if locator.count():
            text = locator.inner_text(timeout=1_000).strip()
            if text:
                return text[-4000:]
    except Exception:
        pass
    return "(debug log unavailable or empty)"


def wait_js(page, expression: str, timeout: int = SEARCH_TIMEOUT_MS) -> bool:
    try:
        page.wait_for_function(expression, timeout=timeout)
        return True
    except PlaywrightTimeoutError:
        return False


def set_analysis_depth(page) -> None:
    options = page.locator("#an-depth option").evaluate_all(
        "(els) => els.map(e => e.value)"
    )
    depth = ANALYSIS_DEPTH if ANALYSIS_DEPTH in options else options[0]
    page.select_option("#an-depth", depth)


def ensure_analysis_engine_on(page) -> None:
    button = page.locator("#an-eng-btn")
    text = button.inner_text().strip().upper()
    if text != "STOP":
        button.click()


def ensure_analysis_engine_off(page) -> None:
    button = page.locator("#an-eng-btn")
    text = button.inner_text().strip().upper()
    if text == "STOP":
        button.click()


def test_clock_visibility(page) -> None:
    print("\n--- B2: Clock visibility ---")
    page.select_option("#sel-mode", "game")
    check(
        "Clock-top visible in Game mode",
        page.evaluate(
            "window.getComputedStyle(document.getElementById('clock-top')).display !== 'none'"
        ),
    )
    check(
        "Clock-bot visible in Game mode",
        page.evaluate(
            "window.getComputedStyle(document.getElementById('clock-bot')).display !== 'none'"
        ),
    )
    check(
        "Clock-top has time text",
        ":" in page.locator("#clock-top").inner_text(),
    )
    check(
        "Clock-bot has time text",
        ":" in page.locator("#clock-bot").inner_text(),
    )

    page.select_option("#sel-mode", "depth")
    check(
        "Clock-top hidden in Depth mode",
        page.evaluate(
            "window.getComputedStyle(document.getElementById('clock-top')).display === 'none'"
        ),
    )


def test_analysis_pv(page) -> None:
    print("\n--- B3: Analysis PV ---")
    page.evaluate("switchTab('analysis')")
    set_analysis_depth(page)
    ensure_analysis_engine_on(page)

    ready = wait_js(
        page,
        """() => {
            const pv = document.getElementById('an-lm1')?.textContent?.trim() || '';
            const sc = document.getElementById('an-ls1')?.textContent?.trim() || '';
            return pv !== '' && pv !== '—' && sc !== '';
        }""",
    )

    pv = page.locator("#an-lm1").inner_text().strip()
    score = page.locator("#an-ls1").inner_text().strip()
    check(
        "PV line 1 has content",
        ready and bool(pv) and pv != "—",
        f"PV={pv!r}; debug={debug_log(page)}",
    )
    check(
        "Score is displayed",
        ready and bool(score),
        f"score={score!r}; debug={debug_log(page)}",
    )

    SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)
    page.screenshot(path=str(SCREENSHOT_DIR / "analysis_pv.png"))
    ensure_analysis_engine_off(page)


def test_multipv(page) -> None:
    print("\n--- B4: MultiPV ---")
    page.evaluate("switchTab('analysis')")
    set_analysis_depth(page)

    # Return to a known one-PV state before starting.
    while page.locator("#an-pv-count").inner_text().strip() != "1":
        page.evaluate("anPvMinus()")

    ensure_analysis_engine_on(page)
    first_ready = wait_js(
        page,
        "() => { const t=document.getElementById('an-lm1')?.textContent?.trim()||''; return t && t !== '—'; }",
    )
    check(
        "Initial PV line produced",
        first_ready,
        debug_log(page),
    )

    page.evaluate("anPvPlus()")
    count_ready = wait_js(
        page,
        "() => document.getElementById('an-pv-count')?.textContent?.trim() === '2'",
        timeout=5_000,
    )
    check("PV count increased to 2", count_ready)

    line2_ready = wait_js(
        page,
        """() => {
            const row = document.getElementById('an-line2');
            const text = document.getElementById('an-lm2')?.textContent?.trim() || '';
            return row && getComputedStyle(row).display !== 'none' &&
                   text !== '' && text !== '—';
        }""",
    )

    line2_display = page.evaluate(
        "window.getComputedStyle(document.getElementById('an-line2')).display"
    )
    line2_text = page.locator("#an-lm2").inner_text().strip()
    check("Line 2 visible", line2_display != "none", f"display={line2_display}")
    check(
        "Line 2 has move text",
        line2_ready and bool(line2_text) and line2_text != "—",
        f"line2={line2_text!r}; debug={debug_log(page)}",
    )

    SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)
    page.screenshot(path=str(SCREENSHOT_DIR / "multipv.png"))

    ensure_analysis_engine_off(page)
    if page.locator("#an-pv-count").inner_text().strip() != "1":
        page.evaluate("anPvMinus()")


def test_clock_search_without_book(page) -> None:
    print("\n--- B5: Clock mode without opening book ---")
    page.evaluate("switchTab('game')")
    page.select_option("#sel-mode", "game")

    # This directly tests the historical freeze case: force the normal search
    # path instead of depending on how many opening-book plies happen to exist.
    page.evaluate("bookMove = function () { return null; }")
    page.evaluate("newGame()")

    initial_clock = page.locator("#clock-top").inner_text().strip()
    initial_moves = page.evaluate(
        "() => typeof moveHistory !== 'undefined' ? moveHistory.length : -1"
    )
    check("New game starts with empty history", initial_moves == 0, f"moves={initial_moves}")

    page.evaluate("doMove('e2e4')")

    responded = wait_js(
        page,
        "() => typeof moveHistory !== 'undefined' && moveHistory.length >= 2",
        timeout=SEARCH_TIMEOUT_MS,
    )
    final_moves = page.evaluate(
        "() => typeof moveHistory !== 'undefined' ? moveHistory.length : -1"
    )
    current_clock = page.locator("#clock-top").inner_text().strip()

    check(
        "Engine responds in clock mode with book disabled",
        responded and final_moves >= 2,
        f"moveHistory.length={final_moves}; debug={debug_log(page)}",
    )
    check(
        "Engine clock consumed time",
        current_clock != initial_clock,
        f"initial={initial_clock}, current={current_clock}",
    )

    SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)
    page.screenshot(path=str(SCREENSHOT_DIR / "clock_no_book.png"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("legacy_version", nargs="?", default="")
    parser.add_argument("--version", default=VERSION)
    parser.add_argument("--html", default=HTML_FILE)
    parser.add_argument("--port", type=int, default=PORT)
    parser.add_argument("--headless", action="store_true", default=HEADLESS)
    parser.add_argument("--headed", action="store_false", dest="headless")
    parser.add_argument("--viewport-width", type=int, default=VIEWPORT_W)
    parser.add_argument("--viewport-height", type=int, default=VIEWPORT_H)
    parser.add_argument("--ready-timeout-ms", type=int, default=ENGINE_READY_TIMEOUT_MS)
    parser.add_argument("--search-timeout-ms", type=int, default=SEARCH_TIMEOUT_MS)
    return parser.parse_args()


def main() -> int:
    global ENGINE_READY_TIMEOUT_MS, SEARCH_TIMEOUT_MS

    args = parse_args()
    ENGINE_READY_TIMEOUT_MS = args.ready_timeout_ms
    SEARCH_TIMEOUT_MS = args.search_timeout_ms

    version = args.version or args.legacy_version
    directory = engine_dir(version)
    html = directory / args.html
    if not html.is_file():
        print(f"ERROR: browser artifact not found: {html}")
        return 1

    print("=" * 78)
    print(f"Zchezz — Browser/WASM E2E ({directory.name})")
    print("=" * 78)

    httpd = start_server(directory, args.port)
    url = f"http://127.0.0.1:{args.port}/{args.html}"
    print(f"Serving {directory} at {url}")

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=args.headless)
            context = browser.new_context(
                viewport={"width": args.viewport_width, "height": args.viewport_height}
            )
            page = context.new_page()

            console_errors: list[str] = []
            page_errors: list[str] = []
            page.on(
                "console",
                lambda msg: console_errors.append(msg.text)
                if msg.type == "error"
                else None,
            )
            page.on("pageerror", lambda exc: page_errors.append(str(exc)))

            print("Loading page...")
            page.goto(url, wait_until="domcontentloaded", timeout=30_000)

            print("\n--- B1: WASM engine readiness ---")
            runtime_ready = wait_js(
                page,
                "() => typeof window.zchezzSearch === 'function'",
                timeout=ENGINE_READY_TIMEOUT_MS,
            )
            check(
                "window.zchezzSearch becomes available",
                runtime_ready,
                debug_log(page),
            )

            # Do not manufacture dozens of downstream failures when startup
            # itself is broken. Startup is the root gate for every search test.
            if runtime_ready:
                real_console = [
                    item
                    for item in console_errors
                    if "favicon" not in item.lower()
                    and "404" not in item
                    and "fonts.googleapis" not in item.lower()
                ]
                check(
                    "No JavaScript page/console errors on load",
                    not real_console and not page_errors,
                    f"console={real_console[:5]}, page={page_errors[:5]}",
                )

                info = page.locator("#engine-info").inner_text().strip()
                check(
                    "Runtime reports NNUE loaded",
                    "NNUE" in info and "classical eval" not in info,
                    f"engine-info={info!r}; debug={debug_log(page)}",
                )

                test_clock_visibility(page)
                test_analysis_pv(page)
                test_multipv(page)
                test_clock_search_without_book(page)

            browser.close()
    finally:
        httpd.shutdown()
        httpd.server_close()

    total = passed + failed
    print("\n" + "=" * 78)
    print(f"Browser Test Results: {passed}/{total} passed, {failed} failed")
    if errors_list:
        print("Failures:")
        for item in errors_list:
            print(item)
    print("=" * 78)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
