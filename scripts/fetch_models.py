"""Download the model files into models/ and check their SHA-256.

    python scripts/fetch_models.py            # ALIKED + LightGlue ONNX, DM-Count weights
    python scripts/fetch_models.py --export   # also export DM-Count to ONNX (needs torch)

Sources:
- DM-Count (Wang et al., NeurIPS 2020; MIT) weights from the lwcc release v0.1.
- ALIKED and LightGlue ONNX: the files opencv_extra pins for the OpenCV 5 features tests.
"""

from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"

LWCC = "https://github.com/tersekmatija/lwcc_weights/releases/download/v0.1/"
LG = "https://raw.githubusercontent.com/YangGuanyuhan/lightglue_opencv_project/main/model/"
FILES = {
    "aliked-n16rot-top1k-640.onnx": (LG + "aliked-n16rot-top1k-640.onnx",
                                     "b46868de0794ea0f2d878985d99dc17d3639dfab85d852994254a4cdf0cdc0ea"),
    "aliked_lightglue.onnx": (LG + "aliked_lightglue.onnx",
                              "a1bd99d294c4c6c08ca978af3128c85f8c0e603b04a03e4391428455d9624439"),
    "src/DM-Count_SHA.pth": (LWCC + "DM-Count_SHA.pth",
                             "9cd1536a52824ba0102a85659aae46600140c57b6df9cf0356def8514043f556"),
    "src/DM-Count_SHB.pth": (LWCC + "DM-Count_SHB.pth",
                             "12bf4b553e4d795a936f35e0dcbbdfafa34b5d998f80390f7a0ec87a0cf9b6f2"),
    "src/DM-Count_QNRF.pth": (LWCC + "DM-Count_QNRF.pth",
                              "c354cb0d49bdf28f580a74c0ae962721b2eec939f84485fd3f1684f0e36697ee"),
}
# ONNX produced by tools/export_dmcount.py (torch 2.x, opset 17) from the weights above
EXPORTED = {
    "dmcount_sha.onnx": "5e7d112376a3126e575ae49817077825b0cffed519604bff0a199f8479ab5261",
    "dmcount_shb.onnx": "0cd868b8fd96b82ddd2a27229c586498910b10b88aa20e0d6e5932795100b517",
    "dmcount_qnrf.onnx": "85e8760b6e00d0277edd0dd469e76e59a1af4f34f19a46bef9ecbd77aaae6e60",
}


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", action="store_true")
    args = ap.parse_args()
    bad = 0
    for rel, (url, want) in FILES.items():
        dst = MODELS / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists() or sha256(dst) != want:
            print(f"downloading {rel}")
            urllib.request.urlretrieve(url, dst)
        ok = sha256(dst) == want
        bad += not ok
        print(f"{'ok ' if ok else 'BAD'} {rel}")
    if args.export:
        for name in ("sha", "shb", "qnrf"):
            src = MODELS / "src" / f"DM-Count_{name.upper()}.pth"
            subprocess.run([sys.executable, str(ROOT / "tools" / "export_dmcount.py"), str(src),
                            str(MODELS / f"dmcount_{name}.onnx")], check=True)
    for name, want in EXPORTED.items():
        p = MODELS / name
        if p.exists():
            print(f"{'ok ' if sha256(p) == want else 'differs (re-exported)'} {name}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
