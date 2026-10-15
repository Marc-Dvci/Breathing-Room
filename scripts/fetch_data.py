"""Download the Forschungszentrum Jülich pedestrian data used by the console and the tests.

    python scripts/fetch_data.py [--videos all|demo|none]

Everything goes under $BR_DATA (default: ../data next to the repository):

    fzj/videos/   2018crowdqueue overhead videos (DOI 10.34735/ped.2018.1), 1.1 GB for all runs
    fzj/traj/     2018crowdqueue trajectories (metres, 25 fps)
    fzj/ratings/  2022 effects-on-pushing ratings (DOI 10.34735/ped.2022.4)
    fzj/bneck/    2021 bottleneck trajectories (DOI 10.34735/ped.2021.9)

All three datasets are CC BY 4.0. `--videos demo` keeps only run 150, which the console
replays; test A needs runs 030, 040, 050, 060, 150 and 160.
"""

from __future__ import annotations

import argparse
import os
import shutil
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("BR_DATA", ROOT.parent / "data"))
E18 = "http://ped.fz-juelich.de/experiments/2018.01.23_Wuppertal_Entrance/data/"
B21 = "http://ped.fz-juelich.de/experiments/2021.10.08_Duesseldorf_MEH_CroMa/data/2021bottleneck/"
SOURCES = {
    "traj": E18 + "2018crowdqueue/trajectories_txt.zip",
    "ratings": E18 + "2022_effect_on_pushing/2022_effect_on_pushing_rating_data.zip",
    "bneck": B21 + "2021bottleneck_trajecories_txt.zip",  # sic: the archive's own spelling
    "videos": E18 + "2018crowdqueue/videos.zip",
}
DEMO_RUN = "150_q_56_h0.mp4"


def fetch(name: str, url: str, keep=None) -> None:
    zpath = DATA / "fzj" / f"{name}.zip"
    out = DATA / "fzj" / name
    zpath.parent.mkdir(parents=True, exist_ok=True)
    if not zpath.exists():
        print(f"downloading {url}")
        with urllib.request.urlopen(url) as r, open(zpath, "wb") as f:
            shutil.copyfileobj(r, f, 1 << 20)
    out.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zpath) as z:
        for m in z.infolist():
            if m.is_dir() or m.filename.startswith("__MACOSX"):
                continue
            if keep and Path(m.filename).name not in keep:
                continue
            target = out / m.filename
            if not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                with z.open(m) as src, open(target, "wb") as dst:
                    shutil.copyfileobj(src, dst)
    print(f"{name}: {sum(1 for p in out.rglob('*') if p.is_file())} files in {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", choices=["all", "demo", "none"], default="demo")
    args = ap.parse_args()
    for name in ("traj", "ratings", "bneck"):
        fetch(name, SOURCES[name])
    if args.videos != "none":
        fetch("videos", SOURCES["videos"], keep=None if args.videos == "all" else {DEMO_RUN})


if __name__ == "__main__":
    main()
