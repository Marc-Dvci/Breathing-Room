"""Loaders for the Forschungszentrum Jülich pedestrian dynamics data archive.

- 2018crowdqueue (DOI 10.34735/ped.2018.1): entrance experiments, overhead video, PeTrack
  trajectories in metres at 25 fps. Run names encode run_priming_width_motivation,
  e.g. 110_c_12_h0 (h0 = high motivation, h- = low).
- 2022effects_on_pushing (DOI 10.34735/ped.2022.4): per-participant ratings by trained raters,
  1 falling behind, 2 walking, 3 mild pushing, 4 strong pushing, 0 not in video.
- 2021bottleneck (DOI 10.34735/ped.2021.9): bottleneck runs, widths in the file name (w080 = 0.80 m).

All data CC BY 4.0. Download with scripts/fetch_data.py.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

FPS = 25.0

RATING_NAMES = {0: "not in video", 1: "falling behind", 2: "walking", 3: "mild pushing", 4: "strong pushing"}


@dataclass
class Trajectories:
    """Head trajectories of one run, sorted by (frame, id)."""

    run: str
    ids: np.ndarray  # (N,) int
    frames: np.ndarray  # (N,) int
    xy: np.ndarray  # (N, 2) metres
    z: np.ndarray  # (N,) metres (head height as tracked)
    fps: float = FPS

    @property
    def n_frames(self) -> int:
        return int(self.frames.max()) + 1

    def at(self, frame: int) -> tuple[np.ndarray, np.ndarray]:
        lo = np.searchsorted(self.frames, frame, "left")
        hi = np.searchsorted(self.frames, frame, "right")
        return self.ids[lo:hi], self.xy[lo:hi]

    def velocities(self, dt_frames: int = 12) -> np.ndarray:
        """Central-difference velocity (m/s) per row over +-dt/2 frames, NaN at track ends."""
        v = np.full_like(self.xy, np.nan)
        order = np.lexsort((self.frames, self.ids))
        ids, fr, xy = self.ids[order], self.frames[order], self.xy[order]
        h = dt_frames // 2
        starts = np.r_[0, np.flatnonzero(np.diff(ids)) + 1]
        ends = np.r_[starts[1:], len(ids)]
        out = np.full_like(xy, np.nan)
        for s, e in zip(starts, ends):
            n = e - s
            if n <= 2 * h:
                continue
            seg = xy[s:e]
            dfr = (fr[s + 2 * h:e] - fr[s:e - 2 * h]).astype(float)
            out[s + h:e - h] = (seg[2 * h:] - seg[:-2 * h]) / (dfr[:, None] / self.fps)
        v[order] = out
        return v


def parse_run_name(name: str) -> dict:
    m = re.match(r"(\d{3})_([cq])_(\d{2})_(h0|h-|h\+)", name)
    if not m:
        return {"run": name}
    run, priming, width, mot = m.groups()
    return {
        "run": run,
        "priming": {"c": "competition", "q": "queue"}[priming],
        "corridor_width_m": int(width) / 10.0,
        "motivation": {"h0": "high", "h-": "low", "h+": "very high"}[mot],
    }


def load_trajectories(path: str | Path) -> Trajectories:
    path = Path(path)
    data = np.loadtxt(path, comments="#", ndmin=2)
    ids = data[:, 0].astype(np.int64)
    frames = data[:, 1].astype(np.int64)
    order = np.lexsort((ids, frames))
    return Trajectories(path.stem, ids[order], frames[order], data[order, 2:4], data[order, 4])


def load_ratings(path: str | Path, n_frames: int | None = None) -> dict[int, np.ndarray]:
    """Per-person rating at every frame, from the change points in the header.

    Returns {person id: int array of length n_frames}, 0 where the person is not rated."""
    lines = Path(path).read_text(encoding="utf-8", errors="replace").splitlines()
    changes: dict[int, list[tuple[int, int | None]]] = {}
    cur = None
    pat = re.compile(r"#\s*(\d+)?\s*\|\s*Frame\s+(\d+)\s*:\s*(\w+)")
    max_frame = 0
    for ln in lines:
        if not ln.startswith("#"):
            continue
        m = pat.match(ln)
        if not m:
            continue
        pid, fr, val = m.groups()
        if pid:
            cur = int(pid)
        if cur is None:
            continue
        fr = int(fr)
        max_frame = max(max_frame, fr)
        changes.setdefault(cur, []).append((fr, None if val.lower() == "end" else int(val)))
    n = n_frames or max_frame + 1
    out = {}
    for pid, ch in changes.items():
        arr = np.zeros(n, np.int8)
        ch = sorted(ch, key=lambda c: c[0])
        for (f0, val), nxt in zip(ch, ch[1:] + [(n, None)]):
            if val is not None:
                arr[f0:min(nxt[0], n)] = val
        out[pid] = arr
    return out


def bottleneck_width(name: str) -> float | None:
    m = re.search(r"_w(\d{3})_", name)
    return int(m.group(1)) / 100.0 if m else None


def bottleneck_motivation(name: str) -> str | None:
    m = re.search(r"_(h\w+?)(?:_|$)", Path(name).stem)
    return m.group(1) if m else None
