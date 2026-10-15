"""Per-stage timing of the Breathing Room camera pipeline, for COOL against pip OpenCV 5.

Run the same script, on the same instance, with two interpreters:

    /opt/cool/.../python bench/bench_stages.py --label cool  --out bench/results/cool_c8g.json
    .venv/bin/python     bench/bench_stages.py --label pip   --out bench/results/pip_c8g.json

Each stage runs on the same recorded frames (Jülich entrance run 150), with warm-up, and
reports the median and p90 in milliseconds. `cameras_realtime` converts the per-second work
of one camera (5 field samples per second, DM-Count every 5 s) into how many cameras one
instance keeps up with in real time.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def imds(path: str) -> str | None:
    try:
        tok = urllib.request.urlopen(urllib.request.Request("http://169.254.169.254/latest/api/token", method="PUT",
                                                            headers={"X-aws-ec2-metadata-token-ttl-seconds": "60"}), timeout=1).read().decode()
        return urllib.request.urlopen(urllib.request.Request(f"http://169.254.169.254/latest/meta-data/{path}",
                                                             headers={"X-aws-ec2-metadata-token": tok}), timeout=1).read().decode()
    except Exception:
        return None


def timeit(fn, n: int, warm: int = 3) -> dict:
    for _ in range(warm):
        fn()
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t0) * 1000)
    return {"median_ms": float(np.median(ts)), "p90_ms": float(np.percentile(ts, 90)), "n": n}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--video", default=str(Path(os.environ.get("BR_DATA", ROOT.parent / "data")) / "fzj/videos/150_q_56_h0.mp4"))
    ap.add_argument("--calib", default=str(ROOT / "eval/calib/fzj_2018crowdqueue.json"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--threads", type=int, default=0, help="cv2.setNumThreads; 0 keeps the default")
    args = ap.parse_args()
    if args.threads:
        cv2.setNumThreads(args.threads)

    import sys

    sys.path.insert(0, str(ROOT / "src"))
    from breathingroom.datasets.fzj_camera import detect_caps, load
    from breathingroom.density import DensityModel
    from breathingroom.drift import DriftWatch

    lens, runs = load(args.calib)
    cap = cv2.VideoCapture(args.video)
    frames = []
    for f in (300, 302, 600):
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        frames.append(cap.read()[1])
    a, b, c = frames
    H, W = a.shape[:2]
    m1, m2 = cv2.initUndistortRectifyMap(lens.K, lens.D, None, lens.Kn, (W, H), cv2.CV_16SC2)
    ua, ub = cv2.remap(a, m1, m2, cv2.INTER_LINEAR), cv2.remap(b, m1, m2, cv2.INTER_LINEAR)
    ga = cv2.resize(cv2.cvtColor(ua, cv2.COLOR_BGR2GRAY), (960, 720), interpolation=cv2.INTER_AREA)
    gb = cv2.resize(cv2.cvtColor(ub, cv2.COLOR_BGR2GRAY), (960, 720), interpolation=cv2.INTER_AREA)
    dis = cv2.DISOpticalFlow_create(cv2.DISOpticalFlow_PRESET_MEDIUM)
    Hw = np.array(runs["150"]["H_world_to_undist"])
    heat = (np.random.default_rng(0).random((95, 70)) * 255).astype(np.uint8)
    dm_new = DensityModel("sha", "new", max_side=960)
    dm_ort = DensityModel("sha", "ort", max_side=960) if hasattr(cv2.dnn, "ENGINE_ORT") else None

    stages = {
        "remap_undistort_1920x1440": lambda: cv2.remap(a, m1, m2, cv2.INTER_LINEAR),
        "cvtColor_bgr2gray_1920x1440": lambda: cv2.cvtColor(ua, cv2.COLOR_BGR2GRAY),
        "resize_area_to_960": lambda: cv2.resize(ua, (960, 720), interpolation=cv2.INTER_AREA),
        "cvtColor_bgr2hsv_1920x1440": lambda: cv2.cvtColor(a, cv2.COLOR_BGR2HSV),
        "head_detection_hsv_morph_components": lambda: detect_caps(a),
        "dis_flow_medium_960x720": lambda: dis.calc(ga, gb, None),
        "warpPerspective_overlay_960x720": lambda: cv2.warpPerspective(heat, Hw, (960, 720)),
        "gaussianBlur_fields_100x100": lambda: cv2.GaussianBlur(np.ones((100, 100), np.float32), (0, 0), 5),
        "jpeg_encode_960x720": lambda: cv2.imencode(".jpg", cv2.resize(ua, (960, 720)), [cv2.IMWRITE_JPEG_QUALITY, 80]),
        "dmcount_sha_dnn_new_960x720": lambda: dm_new(ua),
    }
    if dm_ort is not None:
        stages["dmcount_sha_dnn_ort_960x720"] = lambda: dm_ort(ua)
    try:
        dw = DriftWatch(a, np.eye(3), np.array([[100.0, 100.0]]))
        stages["drift_check_aliked_lightglue_640"] = lambda: dw.check(c)
    except Exception as e:  # models missing on a bare install
        print("drift stage skipped:", e)

    res = {}
    for name, fn in stages.items():
        n = max(5, args.n // 5) if ("dmcount" in name or "drift" in name) else args.n
        res[name] = timeit(fn, n)
        print(f"{name:42s} {res[name]['median_ms']:9.2f} ms")

    # per camera-second: 5 field samples x (2 remaps, 2 gray+resize, heads, DIS) + overlay + jpeg,
    # DM-Count once every 5 s, a drift check once a minute
    m = {k: v["median_ms"] for k, v in res.items()}
    per_cam_ms = 5 * (2 * m["remap_undistort_1920x1440"] + 2 * (m["cvtColor_bgr2gray_1920x1440"] + m["resize_area_to_960"])
                      + m["head_detection_hsv_morph_components"] + m["dis_flow_medium_960x720"]) \
        + m["warpPerspective_overlay_960x720"] + m["jpeg_encode_960x720"] + m["dmcount_sha_dnn_new_960x720"] / 5 \
        + m.get("drift_check_aliked_lightglue_640", 0) / 60
    out = {
        "label": args.label,
        "opencv_version": cv2.__version__,
        "opencv_module": cv2.__file__,
        "build_head": cv2.getBuildInformation().splitlines()[:12],
        "threads": cv2.getNumThreads(),
        "cpu": platform.processor() or platform.machine(),
        "machine": platform.machine(),
        "instance_type": imds("instance-type"),
        "ami_id": imds("ami-id"),
        "stages": res,
        "camera_second_ms": per_cam_ms,
        "cameras_realtime": 1000.0 / per_cam_ms,
        "env_cool": {k: v for k, v in os.environ.items() if "COOL" in k.upper()},
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"one camera-second of work: {per_cam_ms:.0f} ms -> {out['cameras_realtime']:.2f} cameras in real time")


if __name__ == "__main__":
    main()
