#!/usr/bin/env python3
"""browser_test.py — Automated browser tests for Zchezz HTML/WASM.

Tests:
  Bug 3: Clocks visible in Game mode, hidden in Depth mode
  Bug 1: Analysis PV shows full line (not just first move)
  Bug 2: MultiPV + button shows 2nd/3rd lines
  Bug 4: Clock mode doesn't freeze after book exhausts

Usage:
  python tests/browser_test.py [version]    # e.g., python tests/browser_test.py v305
"""

import sys, os, time, threading, http.server, socketserver, io, functools

if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import glob
from playwright.sync_api import sync_playwright

def find_latest_engine():
    base = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "engine", "c")
    dirs = sorted(glob.glob(os.path.join(base, "zchezz_v*")))
    if not dirs:
        raise FileNotFoundError("No engine version found")
    return dirs[-1]

if len(sys.argv) > 1 and sys.argv[1].startswith("v"):
    ENGINE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                              "engine", "c", f"zchezz_{sys.argv[1]}")
else:
    ENGINE_DIR = find_latest_engine()
HTML_FILE = "zchezz_wasm.html"
PORT = 8766  # different port to avoid conflicts

passed = 0
failed = 0
errors_list = []
SCREENSHOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "elo_results")
os.makedirs(SCREENSHOT_DIR, exist_ok=True)


def check(name, condition, detail=""):
    global passed, failed
    if condition:
        passed += 1
        print(f"  PASS: {name}")
    else:
        failed += 1
        msg = f"  FAIL: {name}" + (f" -- {detail}" if detail else "")
        print(msg)
        errors_list.append(msg)


def start_server():
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=ENGINE_DIR)
    httpd = socketserver.TCPServer(("127.0.0.1", PORT), handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    return httpd


def test_bug3_clocks_visible(page):
    """Bug 3: Clocks should be visible when Game mode selected."""
    print("\n--- Bug 3: Clock Visibility in Game Mode ---")

    # Select "game" from the mode dropdown
    # The dropdown is <select id="sel-mode"> with options: depth, time, game
    page.select_option("#sel-mode", "game")
    page.wait_for_timeout(1000)

    # Check clock visibility
    top_display = page.evaluate("window.getComputedStyle(document.getElementById('clock-top')).display")
    bot_display = page.evaluate("window.getComputedStyle(document.getElementById('clock-bot')).display")

    check("Clock-top visible in Game mode",
          top_display != "none",
          f"computed display={top_display}")
    check("Clock-bot visible in Game mode",
          bot_display != "none",
          f"computed display={bot_display}")

    # Check clock text
    top_text = page.evaluate("document.getElementById('clock-top').textContent")
    bot_text = page.evaluate("document.getElementById('clock-bot').textContent")
    check("Clock-top has time text",
          ":" in top_text,
          f"text='{top_text}'")
    check("Clock-bot has time text",
          ":" in bot_text,
          f"text='{bot_text}'")


def test_bug3_clocks_hidden(page):
    """Bug 3b: Clocks should be hidden when depth mode is selected."""
    print("\n--- Bug 3b: Clocks Hidden in Depth Mode ---")

    page.select_option("#sel-mode", "depth")
    page.wait_for_timeout(500)

    top_display = page.evaluate("window.getComputedStyle(document.getElementById('clock-top')).display")
    check("Clock-top hidden in Depth mode",
          top_display == "none",
          f"computed display={top_display}")


def test_bug1_pv_display(page):
    """Bug 1: Analysis PV should show full line, not just first move."""
    print("\n--- Bug 1: Analysis PV Display ---")

    # Switch to Analysis tab
    page.evaluate("switchTab('analysis')")
    page.wait_for_timeout(1000)

    # Toggle engine ON via the ENG button (id=an-eng-btn)
    eng_btn = page.locator("#an-eng-btn")
    if eng_btn.count() > 0:
        eng_btn.click()
        # Wait for engine to produce results
        page.wait_for_timeout(8000)

        # Read PV line 1 text from #an-lm1
        pv_text = page.evaluate("document.getElementById('an-lm1').textContent").strip()

        # PV should have multiple space-separated moves
        if pv_text and pv_text != "—":
            words = pv_text.split()
            check("PV line 1 shows multiple moves",
                  len(words) >= 2,
                  f"PV='{pv_text}' ({len(words)} words)")
        else:
            check("PV line 1 has content",
                  False,
                  f"PV text='{pv_text}'")

        # Check score is displayed
        score_text = page.evaluate("document.getElementById('an-ls1').textContent").strip()
        check("Score is displayed",
              len(score_text) > 0,
              f"score='{score_text}'")

        # Take screenshot
        page.screenshot(path=os.path.join(SCREENSHOT_DIR, "bug1_pv_display.png"))

        # Turn engine off
        eng_btn.click()
        page.wait_for_timeout(500)
    else:
        check("Engine toggle button (#an-eng-btn) found", False)


def test_bug2_multipv(page):
    """Bug 2: MultiPV + button should show 2nd/3rd lines."""
    print("\n--- Bug 2: MultiPV Button ---")

    # Make sure we're on Analysis tab
    page.evaluate("switchTab('analysis')")
    page.wait_for_timeout(500)

    # Start engine
    eng_btn = page.locator("#an-eng-btn")
    if eng_btn.count() > 0:
        eng_btn.click()
        page.wait_for_timeout(5000)

        # Check initial PV count
        initial = page.evaluate("document.getElementById('an-pv-count').textContent").strip()
        check("Initial PV count is 1", initial == "1", f"initial={initial}")

        # Click the + button (onclick=anPvPlus)
        page.evaluate("anPvPlus()")
        page.wait_for_timeout(6000)

        # Check PV count increased
        new_count = page.evaluate("document.getElementById('an-pv-count').textContent").strip()
        check("PV count increased to 2", new_count == "2", f"count={new_count}")

        # Check 2nd line is visible
        line2_display = page.evaluate("window.getComputedStyle(document.getElementById('an-line2')).display")
        check("Line 2 visible", line2_display != "none", f"display={line2_display}")

        # Check 2nd line has content
        lm2_text = page.evaluate("document.getElementById('an-lm2').textContent").strip()
        check("Line 2 has move text",
              len(lm2_text) > 0 and lm2_text != "—",
              f"lm2='{lm2_text}'")

        # Take screenshot
        page.screenshot(path=os.path.join(SCREENSHOT_DIR, "bug2_multipv.png"))

        # Reset and stop
        page.evaluate("anPvMinus()")
        page.wait_for_timeout(500)
        eng_btn.click()
        page.wait_for_timeout(500)
    else:
        check("Engine toggle button found", False)


def test_bug4_clock_game(page):
    """Bug 4: Clock mode should not freeze — engine should keep responding."""
    print("\n--- Bug 4: Clock Mode Game Play ---")

    # Switch back to Game tab
    page.evaluate("switchTab('game')")
    page.wait_for_timeout(500)

    # Select game mode (clock)
    page.select_option("#sel-mode", "game")
    page.wait_for_timeout(500)

    # Start a new game
    page.evaluate("newGame()")
    page.wait_for_timeout(2000)

    # Read initial clock
    initial_clock = page.evaluate("document.getElementById('clock-top').textContent").strip()
    print(f"  (Initial clock: {initial_clock})")

    # Make the human's first move (e2->e4) to trigger the engine
    page.evaluate("doMove('e2e4')")
    page.wait_for_timeout(3000)

    # Play multiple rounds: human move, wait for engine
    human_moves = ['d2d4', 'g1f3', 'f1e2', 'e1g1', 'c2c4', 'b1c3']
    for i, mv in enumerate(human_moves):
        mc = page.evaluate("typeof moveHistory !== 'undefined' ? moveHistory.length : 0")
        if mc < 2 + i * 2:
            break  # Engine didn't respond yet
        try:
            page.evaluate(f"doMove('{mv}')")
        except:
            break  # Move may be illegal
        page.wait_for_timeout(3000)  # Wait for engine response

    # Check: has the game progressed?
    move_count = page.evaluate("typeof moveHistory !== 'undefined' ? moveHistory.length : 0")
    print(f"  (Final moveHistory.length={move_count})")

    check("Game progressed (moves played after book exhausts)",
          move_count > 4,
          f"moveHistory.length={move_count}")

    # Check clock has changed (time was consumed)
    current_clock = page.evaluate("document.getElementById('clock-top').textContent").strip()
    check("Clock time changed (engine was thinking)",
          current_clock != initial_clock,
          f"initial={initial_clock}, current={current_clock}")

    # Take screenshot
    page.screenshot(path=os.path.join(SCREENSHOT_DIR, "bug4_clock_game.png"))


def main():
    print("=" * 60)
    print(f"Zchezz — Automated Browser Tests ({os.path.basename(ENGINE_DIR)})")
    print("=" * 60)

    if not os.path.exists(os.path.join(ENGINE_DIR, HTML_FILE)):
        print(f"ERROR: {HTML_FILE} not found in {ENGINE_DIR}")
        sys.exit(1)

    httpd = start_server()
    url = f"http://127.0.0.1:{PORT}/{HTML_FILE}"
    print(f"\nServing {ENGINE_DIR} at {url}")

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        context = browser.new_context(viewport={"width": 1400, "height": 950})
        page = context.new_page()

        # Capture console errors
        js_errors = []
        page.on("console", lambda msg: js_errors.append(msg.text) if msg.type == "error" else None)

        print(f"Loading page...")
        page.goto(url, timeout=30000)
        page.wait_for_timeout(4000)  # Wait for WASM to compile + init
        print("Page loaded!\n")

        # Filter out harmless errors (favicon 404, resource loads)
        real_errors = [e for e in js_errors if 'favicon' not in e.lower() and '404' not in e]
        check("No JS console errors on load",
              len(real_errors) == 0,
              f"errors: {real_errors[:3]}")

        # Run tests
        test_bug3_clocks_visible(page)
        test_bug3_clocks_hidden(page)
        test_bug1_pv_display(page)
        test_bug2_multipv(page)
        test_bug4_clock_game(page)

        # Final screenshot
        page.screenshot(path=os.path.join(SCREENSHOT_DIR, "browser_test_final.png"))

        browser.close()

    httpd.shutdown()

    # Summary
    total = passed + failed
    print(f"\n{'='*60}")
    print(f"Browser Test Results: {passed}/{total} passed, {failed} failed")
    if errors_list:
        print(f"\nFailures:")
        for e in errors_list:
            print(e)
    print(f"{'='*60}")

    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
