"""Per-stage timing of the Breathing Room camera pipeline, on the machine it runs on.

    python bench/bench_stages.py --label graviton2-t4g.small --out bench/results/graviton2.json

Each stage runs on the same recorded frames (Jülich entrance run 150), with warm-up, and
reports the median and p90 in milliseconds. The light stages, DM-Count and the drift check
each run in their own process, so one model's memory never slows another. DM-Count is timed on
both OpenCV 5 DNN engines (ENGINE_NEW and ENGINE_CLASSIC). Three totals turn the work of one
camera-second into cameras kept in real time per machine: the pipeline at full size, the
console's pipeline at half size (5 field samples per second, overlay, JPEG, the ECC screen every
3 s), and the same with DM-Count every 5 s on the faster engine, as an oblique street camera would
count. The ALIKED + LightGlue check runs on its own schedule and is reported per call.
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
    ap.add_argument("--group", default="all", choices=["all", "light", "dmcount", "drift"])
    args = ap.parse_args()
    if args.threads:
        cv2.setNumThreads(args.threads)

    import sys

    sys.path.insert(0, str(ROOT / "src"))
    from breathingroom.datasets.fzj_camera import detect_caps, load
    from breathingroom.density import DensityModel
    from breathingroom.drift import DriftWatch, ShiftScreen
    from breathingroom.runtime import CameraZone

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
    g = args.group
    stages = {}
    if g in ("light", "dmcount", "drift"):
        cz = CameraZone(Path(args.video), Path(args.calib), "150")  # the console's own camera code
        small = cv2.resize(a, cz.work_size, interpolation=cv2.INTER_AREA)
        u_half, _ = cz.prepare(a)
    if g == "light":
        stages = {
            # full size: the pipeline before the working scale
            "remap_undistort_1920x1440": lambda: cv2.remap(a, m1, m2, cv2.INTER_LINEAR),
            "cvtColor_bgr2gray_1920x1440": lambda: cv2.cvtColor(ua, cv2.COLOR_BGR2GRAY),
            "resize_area_to_960": lambda: cv2.resize(ua, (960, 720), interpolation=cv2.INTER_AREA),
            "head_detection_hsv_morph_components": lambda: detect_caps(a),
            # half size: what the console runs
            "resize_area_frame_to_half": lambda: cv2.resize(a, cz.work_size, interpolation=cv2.INTER_AREA),
            "remap_undistort_960x720": lambda: cv2.remap(small, cz.m1, cz.m2, cv2.INTER_LINEAR),
            "head_detection_half": lambda: detect_caps(small, scale=0.5),
            "cvtColor_bgr2gray_960x720": lambda: cv2.cvtColor(u_half, cv2.COLOR_BGR2GRAY),
            "dis_flow_medium_960x720": lambda: dis.calc(ga, gb, None),
            "warpPerspective_overlay_960x720": lambda: cv2.warpPerspective(heat, Hw, (960, 720)),
            "gaussianBlur_fields_100x100": lambda: cv2.GaussianBlur(np.ones((100, 100), np.float32), (0, 0), 5),
            "jpeg_encode_960x720": lambda: cv2.imencode(".jpg", u_half, [cv2.IMWRITE_JPEG_QUALITY, 80]),
        }
    elif g == "dmcount":
        crop, mask = cz.count_crop(u_half)
        for eng in ("new", "classic"):
            dm = DensityModel("sha", eng, max_side=4096)
            stages[f"dmcount_sha_{eng}_zone_crop"] = (lambda dm=dm: dm(crop, roi_mask=mask))
            dmf = DensityModel("sha", eng, max_side=960)
            stages[f"dmcount_sha_{eng}_960x720"] = (lambda dmf=dmf: dmf(ua))
    elif g == "drift":
        sc = ShiftScreen(cz.screen_frames, cz.to_floor, cz.probe_px())
        stages["camera_screen_ecc_480"] = lambda: sc.check(c)
        dw = DriftWatch(a, np.eye(3), np.array([[100.0, 100.0]]))
        stages["drift_check_aliked_lightglue_640"] = lambda: dw.check(c)

    res = {}
    for name, fn in stages.items():
        n = max(5, args.n // 5) if ("dmcount" in name or "drift" in name) else args.n
        res[name] = timeit(fn, n)
        print(f"{name:42s} {res[name]['median_ms']:9.2f} ms", flush=True)
    if g != "all":
        Path(args.out).write_text(json.dumps(res))
        return

    import subprocess
    import tempfile

    res = {}
    for grp in ("light", "dmcount", "drift"):
        part = Path(tempfile.gettempdir()) / f"bench_{grp}.json"
        subprocess.run([sys.executable, __file__, "--label", args.label, "--out", str(part), "--n", str(args.n),
                        "--threads", str(args.threads), "--group", grp, "--video", args.video, "--calib", args.calib], check=True)
        res.update(json.loads(part.read_text()))
    m = {k: v["median_ms"] for k, v in res.items()}
    full_ms = 5 * (2 * m["remap_undistort_1920x1440"] + 2 * (m["cvtColor_bgr2gray_1920x1440"] + m["resize_area_to_960"])
                   + m["head_detection_hsv_morph_components"] + m["dis_flow_medium_960x720"])         + m["warpPerspective_overlay_960x720"] + m["jpeg_encode_960x720"]
    console_ms = 5 * (2 * (m["resize_area_frame_to_half"] + m["remap_undistort_960x720"] + m["cvtColor_bgr2gray_960x720"])
                      + m["head_detection_half"] + m["dis_flow_medium_960x720"])         + m["warpPerspective_overlay_960x720"] + m["jpeg_encode_960x720"] + m["camera_screen_ecc_480"] / 3
    best_dm = min(m["dmcount_sha_new_zone_crop"], m["dmcount_sha_classic_zone_crop"])
    counting_ms = console_ms + best_dm / 5
    out = {
        "label": args.label,
        "opencv_version": cv2.__version__,
        "build_head": cv2.getBuildInformation().splitlines()[:12],
        "threads": cv2.getNumThreads(),
        "cpu": platform.processor() or platform.machine(),
        "machine": platform.machine(),
        "instance_type": imds("instance-type"),
        "ami_id": imds("ami-id"),
        "stages": res,
        "camera_second_ms": {"console": round(console_ms, 1), "full_size_pipeline": round(full_ms, 1),
                             "with_dmcount_every_5s": round(counting_ms, 1)},
        "cameras_realtime": {"console": round(1000 / console_ms, 2), "with_dmcount_every_5s": round(1000 / counting_ms, 2)},
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"full-size pipeline: {full_ms:.0f} ms; one camera-second: {console_ms:.0f} ms ({out['cameras_realtime']['console']} cameras in real time); "
          f"with DM-Count every 5 s: {counting_ms:.0f} ms ({out['cameras_realtime']['with_dmcount_every_5s']})")


if __name__ == "__main__":
    main()
