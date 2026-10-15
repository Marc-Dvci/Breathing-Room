"""Drive the steward console headless and check what a person would see.

    python tools/ui_smoke.py http://127.0.0.1:8765 --out var/ui_smoke

Steps: load the page, wait for both zone tiles and a camera frame, knock the camera and wait
for the recalibration card (or a trust flag on the E1 tile), restore it, then wait for the
assistant's proposal on Zone C, approve it, and wait for the verify outcome. Screenshots of
each state go to --out. Console errors and page errors fail the run.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright


def chromium_path() -> str | None:
    """An installed headless Chromium when the package's own build is missing."""
    import glob
    import os

    if os.environ.get("PW_CHROMIUM"):
        return os.environ["PW_CHROMIUM"]
    base = os.path.join(os.environ.get("LOCALAPPDATA", ""), "ms-playwright")
    found = sorted(glob.glob(os.path.join(base, "chromium_headless_shell-*", "*", "chrome-headless-shell.exe")))
    return found[-1] if found else None


TWIN_CARD = '.prop[data-zone^="Zone C"]'
TWIN_APPROVE = f"{TWIN_CARD} button.approve"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("url")
    ap.add_argument("--out", default="var/ui_smoke")
    ap.add_argument("--proposal-timeout", type=float, default=900)
    ap.add_argument("--verify-timeout", type=float, default=300)
    ap.add_argument("--skip-knock", action="store_true")
    ap.add_argument("--skip-approve", action="store_true")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    errors: list[str] = []
    log: list[dict] = []

    def note(step: str, **kw):
        rec = {"t": round(time.time(), 1), "step": step, **kw}
        log.append(rec)
        print(json.dumps(rec), flush=True)

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, executable_path=chromium_path())
        try:
            page = browser.new_page(viewport={"width": 1600, "height": 1000}, device_scale_factor=1)
            page.on("console", lambda m: errors.append(f"console.{m.type}: {m.text}") if m.type == "error" else None)
            page.on("pageerror", lambda e: errors.append(f"pageerror: {e}"))
            page.goto(args.url, wait_until="networkidle")
            page.wait_for_selector(".tile >> nth=1", timeout=30000)
            page.wait_for_function("document.querySelector('#cam').naturalWidth > 0", timeout=30000)
            note("loaded", tiles=page.locator(".tile").count())
            page.screenshot(path=str(out / "01_loaded.png"), full_page=True)

            if not args.skip_knock:
                page.click("#bump")
                note("knocked")
                page.wait_for_selector(".tile.untrusted", timeout=60000)
                note("camera flagged", untrusted=page.locator(".tile.untrusted").count())
                page.wait_for_selector(".recal", timeout=120000)
                note("recalibration requested", text=page.locator(".recal").inner_text())
                time.sleep(2)
                page.screenshot(path=str(out / "02_recalibration.png"), full_page=True)
                page.click("#unbump")
                note("restored")

            t0 = time.time()
            while time.time() - t0 < args.proposal_timeout:
                if page.locator(TWIN_APPROVE).count():
                    break
                time.sleep(2)
            else:
                note("no proposal", waited_s=round(time.time() - t0))
                page.screenshot(path=str(out / "03_no_proposal.png"), full_page=True)
                return 1
            card = page.locator(TWIN_CARD)
            note("proposal", radio=card.locator(".radio").inner_text(),
                 planner=card.locator(".kicker").inner_text(), waited_s=round(time.time() - t0))
            page.screenshot(path=str(out / "03_proposal.png"), full_page=True)

            if not args.skip_approve:
                page.click(TWIN_APPROVE)
                note("approved")
                page.wait_for_selector(f"{TWIN_CARD} .status", timeout=10000)
                page.screenshot(path=str(out / "04_approved.png"), full_page=True)
                page.wait_for_function(
                    "(() => { const s = document.querySelector('.prop[data-zone^=\"Zone C\"] .status');"
                    " return s && /working|escalated/.test(s.textContent); })()",
                    timeout=args.verify_timeout * 1000)
                note("verified", status=page.locator(f"{TWIN_CARD} .status").inner_text())
                page.screenshot(path=str(out / "05_verified.png"), full_page=True)
        finally:
            browser.close()
    (out / "log.json").write_text(json.dumps({"steps": log, "errors": errors}, indent=2))
    if errors:
        print("ERRORS:", *errors, sep="\n  ")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
