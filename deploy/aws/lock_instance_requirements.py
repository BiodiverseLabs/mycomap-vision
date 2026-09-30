"""Write the EC2 instances' lockfiles: every package pinned, with its hashes.

    python deploy/aws/lock_instance_requirements.py            # both files
    python deploy/aws/lock_instance_requirements.py trainer    # one

The instances run CPython 3.11 on Amazon Linux 2023 (x86_64). The other lockfiles
in requirements/ are made with pip-compile, which resolves for the machine it runs
on: on Windows it drops every `platform_system == "Linux"` dependency (torch's
CUDA runtime, the nvidia-* wheels) and adds Windows-only ones (colorama). So this
asks pip's own resolver for the instance's wheels (`pip install --dry-run --report
--platform ... --python-version 3.11 --only-binary :all:`, which reads package
metadata only, no wheels), then walks every package's requirements under the
instance's markers: a Linux-only requirement pip skipped is added and the
resolution repeated; a package only another platform needs is dropped. Every file
hash of each pinned version then comes from PyPI, as pip-compile --generate-hashes
does. Needs network access to PyPI.

The trainer's torch comes from PyPI with its CUDA runtime as nvidia-*/cuda-* wheels
(torch 2.14 on PyPI is built for CUDA 13, which needs NVIDIA driver 580 or newer).
The Deep Learning *Base* AMI brings the driver and no torch, so nothing is
reinstalled over a CUDA build of the image's own, and the instance checks that
torch sees the GPU before it starts (aws.TRAINER_USER_DATA). torch and torchvision
are pinned to the versions tested on the laptop (TORCH_PINS); everything else
resolves to the newest release that fits.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import tomllib
import urllib.request
from pathlib import Path

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

REPO = Path(__file__).resolve().parents[2]
# Amazon Linux 2023 has glibc 2.34: every manylinux wheel up to 2_34 runs there. pip
# does not widen an explicit --platform on its own, so each tag is listed.
PLATFORM = [a for g in range(34, 4, -1) for a in ("--platform", f"manylinux_2_{g}_x86_64")]
PLATFORM += ["--platform", "manylinux2014_x86_64", "--platform", "manylinux2010_x86_64",
             "--platform", "manylinux1_x86_64", "--platform", "linux_x86_64",
             "--python-version", "3.11", "--implementation", "cp", "--abi", "cp311",
             "--abi", "abi3", "--abi", "none"]
# The marker environment of the instances' python3.11 (Amazon Linux 2023, x86_64).
ENV = {"implementation_name": "cpython", "implementation_version": "3.11.13",
       "os_name": "posix", "platform_machine": "x86_64", "platform_release": "",
       "platform_system": "Linux", "platform_version": "", "python_full_version": "3.11.13",
       "platform_python_implementation": "CPython", "python_version": "3.11",
       "sys_platform": "linux"}
TORCH_PINS = ["torch==2.14.0", "torchvision==0.29.0"]

TARGETS = {
    # file: (extras, extra pins, what it is for)
    "trainer": (["aws", "embed"], TORCH_PINS, "the GPU trainer instance (mv aws-train-job)"),
    "downloader": (["aws"], [], "the photo downloader instance (mv download-photos)"),
}


def wanted(extras: list[str]) -> list[str]:
    project = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    reqs = list(project["dependencies"])
    for e in extras:
        reqs += project["optional-dependencies"][e]
    return reqs


def applies(req: Requirement, extra: str = "") -> bool:
    return req.marker is None or req.marker.evaluate({**ENV, "extra": extra})


def pip_report(reqs: list[str]) -> dict[str, dict]:
    with tempfile.TemporaryDirectory(prefix="vision-trainer-lock-") as tmp:
        report = Path(tmp) / "report.json"
        subprocess.run([sys.executable, "-m", "pip", "install", "--dry-run", "--quiet",
                        "--disable-pip-version-check", "--ignore-installed",
                        "--only-binary", ":all:", *PLATFORM, "--target", str(Path(tmp) / "t"),
                        "--report", str(report), *reqs], check=True)
        items = json.loads(report.read_text(encoding="utf-8"))["install"]
    return {canonicalize_name(i["metadata"]["name"]): i["metadata"] for i in items}


def closure(top: list[str], found: dict[str, dict]) -> tuple[set[str], list[str]]:
    """Names the instance needs, walking requirements under ENV; plus requirements it
    needs that pip didn't resolve (it evaluated the markers for this machine)."""
    needed, missing, queue = set(), [], [(Requirement(r), "") for r in top]
    seen = set()
    while queue:
        req, via_extra = queue.pop()
        if not applies(req, via_extra):
            continue
        name = canonicalize_name(req.name)
        key = (name, frozenset(req.extras))
        if key in seen:
            continue
        seen.add(key)
        needed.add(name)
        meta = found.get(name)
        if meta is None:
            missing.append(str(req).split(";")[0].strip())     # its marker holds here
            continue
        for dep in meta.get("requires_dist", []) or []:
            d = Requirement(dep)
            if d.marker is None:
                queue.append((d, ""))
            elif any(d.marker.evaluate({**ENV, "extra": e}) for e in req.extras):
                queue.append((Requirement(str(d).split(";")[0]), ""))
            elif d.marker.evaluate({**ENV, "extra": ""}):
                queue.append((d, ""))
    return needed, missing


def resolve(top: list[str]) -> list[tuple[str, str]]:
    extra: list[str] = []
    for _ in range(6):
        found = pip_report(top + extra)
        needed, missing = closure(top, found)
        if not missing:
            return sorted((n, found[n]["version"]) for n in needed)
        extra += [m for m in missing if m not in extra]
    raise RuntimeError(f"the resolution did not settle; still missing {missing}")


def hashes(name: str, version: str) -> list[str]:
    url = f"https://pypi.org/pypi/{name}/{version}/json"
    with urllib.request.urlopen(url, timeout=60) as r:
        files = json.load(r)["urls"]
    out = sorted({f["digests"]["sha256"] for f in files})
    if not out:
        raise RuntimeError(f"PyPI lists no files for {name}=={version}")
    return out


def write(target: str) -> Path:
    extras, pins, purpose = TARGETS[target]
    pinned = resolve(wanted(extras) + pins)
    lines = ["#",
             f"# Lockfile for {purpose}: CPython 3.11, Linux x86_64.",
             "# Generated by deploy/aws/lock_instance_requirements.py; do not edit by hand.",
             f"# From pyproject.toml extras {', '.join(extras)}"
             + (f" and {', '.join(pins)}" if pins else "") + ".",
             "#"]
    for name, version in pinned:
        hs = hashes(name, version)
        lines.append(f"{name}=={version} \\")
        lines += [f"    --hash=sha256:{h}" + (" \\" if i < len(hs) - 1 else "")
                  for i, h in enumerate(hs)]
    path = REPO / "requirements" / f"{target}.txt"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"{path}: {len(pinned)} packages")
    return path


if __name__ == "__main__":
    for t in sys.argv[1:] or list(TARGETS):
        write(t)
