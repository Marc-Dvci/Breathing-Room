"""Render the SVG figures to PNG at 2x with headless Chromium."""
import glob
import os
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

HERE = Path(__file__).parent


def chromium():
    base = os.path.join(os.environ.get("LOCALAPPDATA", ""), "ms-playwright")
    found = sorted(glob.glob(os.path.join(base, "chromium_headless_shell-*", "*", "chrome-headless-shell.exe")))
    return found[-1] if found else None


with sync_playwright() as p:
    b = p.chromium.launch(headless=True, executable_path=chromium())
    for svg in sys.argv[1:] or [str(x) for x in HERE.glob("*.svg")]:
        svg = Path(svg)
        text = svg.read_text(encoding="utf-8")
        w = int(text.split('width="')[1].split('"')[0]); h = int(text.split('height="')[1].split('"')[0])
        pg = b.new_page(viewport={"width": w, "height": h}, device_scale_factor=2)
        pg.set_content(f"<html><body style='margin:0;background:#fff'>{text}</body></html>")
        pg.screenshot(path=str(svg.with_suffix(".png")), clip={"x": 0, "y": 0, "width": w, "height": h})
        print(svg.with_suffix(".png"))
    b.close()
