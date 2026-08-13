#!/usr/bin/env python3
"""
fetch.py -- materialise the StatsBomb open-data corpus at a PINNED commit.

Reproducibility contract
------------------------
Every run of this script produces the same raw data, today or in November,
because the archive is addressed by PINNED_SHA rather than by a branch name.
A referee who runs `python fetch.py` gets exactly the match set this paper used.

Why an archive and not `git clone`
----------------------------------
open-data is ~16 GB across ~9,000 large JSON blobs. Both `git fetch --depth 1`
and a blob-filtered partial clone of a specific SHA were measured stalling
indefinitely against GitHub: the server-side pack enumeration for this repo
never starts streaming. The codeload archive for the same commit streams
immediately and reliably. It is addressed by the identical commit SHA, so the
pin is exactly as strong; only the transport differs.

Integrity is then checked against invariants recorded from the pinned tree:
per-directory file counts, which are a property of the commit. A wrong or
drifted download fails loudly rather than silently producing a different N.

Usage
-----
    python fetch.py                    # core + three-sixty (default)
    python fetch.py --include core     # Tier 1 only; skips freeze frames
    python fetch.py --include all      # adds lineups
    python fetch.py --verify           # check an existing tree, download nothing
    python fetch.py --force            # discard and re-materialise
    python fetch.py --keep-archive     # retain the .tar.gz after extraction
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tarfile
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

# --------------------------------------------------------------------------- #
# The pin. Changing this line changes the paper's dataset -- do not edit it
# casually, and re-run the full pipeline and re-state N in the paper if you do.
# --------------------------------------------------------------------------- #
REPO = "statsbomb/open-data"
PINNED_SHA = "b0bc9f22dd77c206ddedc1d742893b3bbe64baec"
ARCHIVE_URL = f"https://codeload.github.com/{REPO}/tar.gz/{PINNED_SHA}"

ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT / "data" / "raw"
DEST = RAW_DIR / "open-data"
CACHE = RAW_DIR / ".cache"
MANIFEST_PATH = RAW_DIR / "MANIFEST.json"

PATH_SETS = {
    "core": ["data/competitions.json", "data/matches/", "data/events/"],
    "360": ["data/three-sixty/"],
    "lineups": ["data/lineups/"],
}
INCLUDE_CHOICES = {
    "core": ["core"],
    "360": ["core", "360"],          # 360 is useless without the events it annotates
    "all": ["core", "360", "lineups"],
}
DEFAULT_INCLUDE = "360"

# File counts at PINNED_SHA, read off the pinned git tree. These are properties
# of the commit: if an extraction does not match, the data is not the pin.
EXPECTED_COUNTS = {
    "data/matches": 80,
    "data/events": 4235,
    "data/three-sixty": 426,
    "data/lineups": 4235,
    "data/competitions.json": 1,
}
TRACKED_DIRS = ["data/matches", "data/events", "data/three-sixty", "data/lineups"]

# Rough extracted size per include set, for the free-space check (bytes).
NEEDED_BYTES = {"core": 14e9, "360": 17.5e9, "all": 17.6e9}


def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if abs(n) < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


# --------------------------------------------------------------------------- #
# download
# --------------------------------------------------------------------------- #
def download(url: str, dest: Path, retries: int = 6) -> Path:
    """Stream `url` to `dest`, resuming a partial file across retries."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    total = None
    attempt = 0

    while attempt <= retries:
        have = dest.stat().st_size if dest.exists() else 0
        req = urllib.request.Request(url, headers={"User-Agent": "press-memory/1.0"})
        if have:
            req.add_header("Range", f"bytes={have}-")
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                if have and resp.status != 206:
                    # server ignored the range; start over
                    dest.unlink(missing_ok=True)
                    have = 0
                length = resp.headers.get("Content-Length")
                if length is not None:
                    total = have + int(length)
                mode = "ab" if have else "wb"
                t0 = time.time()
                last = t0
                with open(dest, mode) as fh:
                    while True:
                        chunk = resp.read(1 << 20)
                        if not chunk:
                            break
                        fh.write(chunk)
                        have += len(chunk)
                        now = time.time()
                        if now - last >= 5:
                            rate = have / max(now - t0, 1e-9)
                            pct = f" ({100 * have / total:.1f}%)" if total else ""
                            print(
                                f"\r[fetch]   {human(have)}{pct} at {human(rate)}/s",
                                end="", flush=True,
                            )
                            last = now
            print(f"\r[fetch]   {human(dest.stat().st_size)} downloaded" + " " * 24)
            return dest
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as exc:
            attempt += 1
            if attempt > retries:
                raise
            wait = min(2 ** attempt, 30)
            print(f"\n[fetch]   transfer interrupted ({exc}); retry {attempt}"
                  f"/{retries} in {wait}s, resuming from {human(have)}")
            time.sleep(wait)
    raise RuntimeError("unreachable")


def wanted(name: str, prefixes: list[str]) -> str | None:
    """Map an archive member to its destination-relative path, or None to skip."""
    parts = name.split("/", 1)
    if len(parts) != 2:
        return None
    rel = parts[1]                       # strip the open-data-<sha>/ root
    for p in prefixes:
        if rel == p or rel.startswith(p):
            return rel
    return None


def extract(archive: Path, prefixes: list[str]) -> None:
    DEST.mkdir(parents=True, exist_ok=True)
    n = 0
    t0 = time.time()
    with tarfile.open(archive, mode="r|gz") as tar:
        for member in tar:
            if not member.isfile():
                continue
            rel = wanted(member.name, prefixes)
            if rel is None:
                continue
            out = (DEST / rel).resolve()
            if not str(out).startswith(str(DEST.resolve())):
                raise RuntimeError(f"unsafe path in archive: {member.name}")
            out.parent.mkdir(parents=True, exist_ok=True)
            src = tar.extractfile(member)
            if src is None:
                continue
            with open(out, "wb") as fh:
                shutil.copyfileobj(src, fh, 1 << 20)
            n += 1
            if n % 500 == 0:
                print(f"\r[fetch]   extracted {n:,} files"
                      f"  ({time.time() - t0:.0f}s)", end="", flush=True)
    print(f"\r[fetch]   extracted {n:,} files in {time.time() - t0:.0f}s" + " " * 12)


# --------------------------------------------------------------------------- #
# manifest and verification
# --------------------------------------------------------------------------- #
def summarise() -> dict:
    out: dict[str, dict] = {}
    for rel in TRACKED_DIRS:
        d = DEST / rel
        if not d.is_dir():
            continue
        files = [p for p in d.rglob("*.json") if p.is_file()]
        out[rel] = {"n_files": len(files),
                    "bytes": sum(p.stat().st_size for p in files)}
    comp = DEST / "data" / "competitions.json"
    if comp.is_file():
        out["data/competitions.json"] = {"n_files": 1, "bytes": comp.stat().st_size}
    return out


def check_counts(contents: dict, include: str) -> list[str]:
    """Compare extracted file counts against the pinned commit's invariants."""
    problems = []
    for rel, info in contents.items():
        exp = EXPECTED_COUNTS.get(rel)
        if exp is not None and info["n_files"] != exp:
            problems.append(
                f"{rel}: found {info['n_files']} files, expected {exp} at "
                f"{PINNED_SHA[:12]}"
            )
    for key in INCLUDE_CHOICES[include]:
        for p in PATH_SETS[key]:
            rel = p.rstrip("/")
            if rel not in contents:
                problems.append(f"{rel}: missing entirely")
    return problems


def write_manifest(include: str, archive_sha: str | None) -> dict:
    contents = summarise()
    manifest = {
        "repo": REPO,
        "pinned_sha": PINNED_SHA,
        "archive_url": ARCHIVE_URL,
        "archive_sha256": archive_sha,
        "include": include,
        "paths": [p for k in INCLUDE_CHOICES[include] for p in PATH_SETS[k]],
        "fetched_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "contents": contents,
    }
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def report(manifest: dict) -> None:
    print(f"\n[fetch] pinned commit : {manifest['pinned_sha']}")
    print(f"[fetch] include set   : {manifest['include']}")
    print(f"[fetch] location      : {DEST}")
    if manifest.get("archive_sha256"):
        print(f"[fetch] archive sha256: {manifest['archive_sha256']}")
    print("[fetch] contents:")
    total = 0
    for rel, info in sorted(manifest["contents"].items()):
        total += info["bytes"]
        exp = EXPECTED_COUNTS.get(rel)
        mark = "" if exp is None or exp == info["n_files"] else f"  != expected {exp}"
        print(f"          {rel:<26} {info['n_files']:>6} files"
              f" {info['bytes'] / 1e6:>9.1f} MB{mark}")
    print(f"          {'TOTAL':<26} {'':>6}       {total / 1e6:>9.1f} MB")
    print(f"\n[fetch] manifest -> {MANIFEST_PATH.relative_to(ROOT)}")


def verify(include: str) -> None:
    if not DEST.exists():
        sys.exit(f"[verify] nothing at {DEST}; run `python fetch.py` first.")
    contents = summarise()
    problems = check_counts(contents, include)
    if problems:
        print("[verify] FAIL: extracted tree does not match the pinned commit:")
        for p in problems:
            print(f"           {p}")
        sys.exit(1)
    print(f"[verify] OK: file counts match {PINNED_SHA[:12]} for include={include}")

    if MANIFEST_PATH.exists():
        recorded = json.loads(MANIFEST_PATH.read_text())
        if recorded.get("pinned_sha") != PINNED_SHA:
            sys.exit(f"[verify] FAIL: manifest records "
                     f"{recorded.get('pinned_sha')}, code pins {PINNED_SHA}.")
        drift = [
            (k, recorded["contents"].get(k), contents.get(k))
            for k in set(recorded["contents"]) | set(contents)
            if recorded["contents"].get(k) != contents.get(k)
        ]
        if drift:
            print("[verify] FAIL: on-disk contents drifted from MANIFEST.json:")
            for k, was, now in drift:
                print(f"           {k}: manifest={was} disk={now}")
            sys.exit(1)
        print("[verify] OK: byte totals match MANIFEST.json")
        report(recorded)
    else:
        print("[verify] no MANIFEST.json; writing one from the current tree.")
        report(write_manifest(include, None))


# --------------------------------------------------------------------------- #
def fetch(include: str, force: bool, keep_archive: bool) -> None:
    prefixes = [p for k in INCLUDE_CHOICES[include] for p in PATH_SETS[k]]

    if DEST.exists() and not force:
        problems = check_counts(summarise(), include)
        if not problems:
            print(f"[fetch] tree already complete for include={include}; "
                  f"nothing to do (use --force to redo).")
            report(write_manifest(include, None))
            return
        print("[fetch] existing tree is incomplete; re-materialising.")

    if DEST.exists():
        print(f"[fetch] removing {DEST}")
        shutil.rmtree(DEST)

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(RAW_DIR).free
    need = NEEDED_BYTES[include]
    print(f"[fetch] free disk {human(free)}; include={include} needs about "
          f"{human(need)} extracted plus ~{human(2e9)} for the archive")
    if free < need + 2e9:
        sys.exit("[fetch] not enough free disk space.")

    archive = CACHE / f"open-data-{PINNED_SHA[:12]}.tar.gz"
    print(f"[fetch] downloading {ARCHIVE_URL}")
    download(ARCHIVE_URL, archive)

    print("[fetch] hashing archive")
    h = hashlib.sha256()
    with open(archive, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 22), b""):
            h.update(block)
    archive_sha = h.hexdigest()

    print(f"[fetch] extracting {include} paths")
    extract(archive, prefixes)

    problems = check_counts(summarise(), include)
    if problems:
        print("\n[fetch] FAIL: extracted tree does not match the pinned commit:")
        for p in problems:
            print(f"          {p}")
        sys.exit(1)
    print(f"[fetch] verified file counts against {PINNED_SHA[:12]}")

    if not keep_archive:
        archive.unlink(missing_ok=True)
        try:
            CACHE.rmdir()
        except OSError:
            pass
        print("[fetch] removed archive (use --keep-archive to retain)")

    report(write_manifest(include, archive_sha))


def main() -> None:
    ap = argparse.ArgumentParser(
        description=f"Fetch StatsBomb open-data pinned at {PINNED_SHA[:12]}."
    )
    ap.add_argument("--include", choices=sorted(INCLUDE_CHOICES),
                    default=DEFAULT_INCLUDE,
                    help="core = competitions+matches+events (Tier 1); "
                         "360 = core plus freeze frames (default); "
                         "all = adds lineups.")
    ap.add_argument("--force", action="store_true", help="discard and re-materialise")
    ap.add_argument("--verify", action="store_true",
                    help="check an existing tree; download nothing")
    ap.add_argument("--keep-archive", action="store_true",
                    help="keep the .tar.gz after extraction")
    args = ap.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    if args.verify:
        verify(args.include)
    else:
        fetch(args.include, args.force, args.keep_archive)


if __name__ == "__main__":
    main()
