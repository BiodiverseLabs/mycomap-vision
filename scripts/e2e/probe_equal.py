"""Is a session's combined answer exactly POST /identify's for the same photos?

  python probe_equal.py <base> --key-file K --photos a.jpg|https://... b.jpg c.jpg [--trials 2]

Sends the photos one at a time into a session and all at once to /identify (prepared as
the app prepares them), and compares species and genus names, scores and confidences to
the last digit. Prints SAME or DIFF per trial and rank; exits 1 on any DIFF.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).parent))
from vision_e2e import app_copy  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("base")
    ap.add_argument("--key-file", required=True)
    ap.add_argument("--photos", nargs="+", required=True)
    ap.add_argument("--trials", type=int, default=2)
    a = ap.parse_args()
    v1 = a.base.rstrip("/") + "/api/v1"
    key = {"Authorization": "Bearer " + Path(a.key_file).read_text().strip()}
    photos = [app_copy(p) for p in a.photos]
    differ = False
    for trial in range(a.trials):
        sid = requests.post(v1 + "/sessions", headers=key, json={}, timeout=60).json()["session_id"]
        for i, body in enumerate(photos):
            r = requests.post(f"{v1}/sessions/{sid}/photos", headers=key, timeout=300,
                              files={"photo": ("p.jpg", body, "image/jpeg")},
                              data={"photo_ref": f"p{i}"}).json()
        session = r["combined"]
        together = requests.post(v1 + "/identify", headers=key, timeout=300, data={"top_k": "5"},
                                 files=[("photos", (f"p{i}.jpg", b, "image/jpeg"))
                                        for i, b in enumerate(photos)]).json()
        for rank in ("species", "genus"):
            s = [(c["name"], c["score"], c["confidence"]) for c in session["ranks"][rank]]
            t = [(c["name"], c["score"], c["confidence"]) for c in together["ranks"][rank]]
            print(trial, rank, "SAME" if s == t else "DIFF")
            if s != t:
                differ = True
                for x, y in zip(s, t):
                    print("   ", x, "|", y)
        requests.delete(f"{v1}/sessions/{sid}", headers=key, timeout=60)
    return 1 if differ else 0


if __name__ == "__main__":
    sys.exit(main())
