"""Push SemiForge to GitHub via Git Data API (git smart protocol is blocked
by the local proxy with 502; gh api works). Idempotent, retry with backoff.

Bootstrap: repo is EMPTY -> first PUT one file via Contents API to create the
initial commit, then blobs -> tree -> commit(parent=bootstrap) -> PATCH main.
"""

from __future__ import annotations

import base64
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

REPO = "CJX0712/semiforge"
ROOT = Path(__file__).resolve().parent.parent  # semiforge repo root
AUTHOR = {"name": "晨星", "email": "CJX0712@users.noreply.github.com"}
COMMIT_MSG = (
    "SemiForge v0.1.0 - modular semi-supervised learning system: calibrated "
    "graph+ensemble fusion (CBCP-Hybrid), class-balanced curriculum "
    "pseudo-labeling with guarded rollback, pure-numpy offline fallback. "
    "38 tests green, deterministic. Author: 晨星"
)
SKIP_DIRS = {".git", ".pytest_cache", ".ruff_cache", "__pycache__", "venv", ".venv"}


def gh(args: list[str], retries: int = 6) -> dict:
    # gh api takes ONE path argument + flags; segments before the first
    # flag are joined with "/", everything from the first "-" arg on is flags
    k = len(args)
    for i, a in enumerate(args):
        if a.startswith("-"):
            k = i
            break
    path = "/".join(args[:k])
    flags = list(args[k:])
    cmd = ["gh", "api", path, *flags]
    for attempt in range(1, retries + 1):
        p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", check=False)
        out, err, code = p.stdout, p.stderr, p.returncode
        transient = code != 0 and any(
            k in (err + out).lower() for k in ("502", "503", "bad gateway", "reset")
        )
        if code == 0:
            return json.loads(out) if out.strip() else {}
        if transient or (code != 0 and attempt < retries and not _permanent(err)):
            wait = min(2**attempt, 30)
            print(f"  retry {attempt} ({err.strip()[:80]}) sleep {wait}s")
            time.sleep(wait)
            continue
        raise RuntimeError(f"gh api failed: {err.strip()[:400]}")
    raise RuntimeError("gh api retries exhausted")


def _permanent(err: str) -> bool:
    return "422" in err or "404" in err or "409" in err


def collect_files() -> dict[str, Path]:
    files: dict[str, Path] = {}
    for p in sorted(ROOT.rglob("*")):
        if p.is_file() and not (set(p.parts) & SKIP_DIRS):
            rel = p.relative_to(ROOT).as_posix()
            if rel == "push_via_api.py":
                continue
            files[rel] = p
    return files


def ensure_blob(sha_cache: dict[str, str], rel: str, path: Path) -> str:
    data = path.read_bytes()
    key = hashlib.sha1(data).hexdigest()
    if key in sha_cache:
        return sha_cache[key]
    r = gh(
        [
            f"repos/{REPO}/git/blobs",
            "-f",
            "content=" + base64.b64encode(data).decode(),
            "-f",
            "encoding=base64",
        ]
    )
    sha_cache[key] = r["sha"]
    return r["sha"]


def build_tree(entries: list[dict]) -> str:
    # nested JSON array must go through --input payload (gh -f mangles it)
    payload = Path(ROOT) / ".git" / "tree_payload.json"
    payload.write_text(json.dumps({"tree": entries}), encoding="utf-8")
    r = gh([f"repos/{REPO}/git/trees", "--input", str(payload)])
    payload.unlink(missing_ok=True)
    return r["sha"]


def main() -> None:
    files = collect_files()
    print(f"{len(files)} files to push")
    # 0) bootstrap initial commit (empty repo -> trees API would 409)
    try:
        head = gh([f"repos/{REPO}/git/refs/heads/main"])
        base_sha = head["object"]["sha"]
        print(f"main exists at {base_sha[:12]}")
    except RuntimeError as e:
        # empty repo: refs GET returns 404 (or 409 "Repository is empty")
        if "404" in str(e) or "409" in str(e):
            base_sha = None
        else:
            raise
    if base_sha is None:
        print("empty repo -> bootstrap via Contents API (README.md)")
        content = (ROOT / "README.md").read_bytes()
        r = gh(
            [
                f"repos/{REPO}/contents/README.md",
                "-X",
                "PUT",
                "-f",
                "message=bootstrap",
                "-f",
                "content=" + base64.b64encode(content).decode(),
            ]
        )
        base_sha = r["commit"]["sha"]
        print(f"bootstrap commit {base_sha[:12]}")

    # 1) blobs
    sha_cache: dict[str, str] = {}
    entries = []
    for rel, path in files.items():
        if rel == "README.md" and base_sha is not None and len(files) > 1:
            pass  # still include (same content -> same blob sha, cheap)
        sha = ensure_blob(sha_cache, rel, path)
        entries.append({"path": rel, "mode": "100644", "type": "blob", "sha": sha})
    print(f"{len(entries)} blobs ready")

    # 2) tree
    tree_sha = build_tree(entries)
    print(f"tree {tree_sha[:12]}")

    # 3) commit (nested JSON via --input payload)
    payload2 = Path(ROOT) / ".git" / "commit_payload.json"
    payload2.write_text(
        json.dumps(
            {
                "message": COMMIT_MSG,
                "tree": tree_sha,
                "parents": [base_sha],
                "author": AUTHOR,
                "committer": AUTHOR,
            }
        ),
        encoding="utf-8",
    )
    commit = gh([f"repos/{REPO}/git/commits", "--input", str(payload2)])
    payload2.unlink(missing_ok=True)
    print(f"commit {commit['sha'][:12]}")

    # 4) PATCH ref
    gh(
        [
            f"repos/{REPO}/git/refs/heads/main",
            "-X",
            "PATCH",
            "-f",
            f"sha={commit['sha']}",
            "-F",
            "force=true",
        ]
    )
    print("refs/heads/main updated")

    # 5) verify
    v = gh([f"repos/{REPO}/commits/main", "--jq", ".sha"])
    print(f"verify head = {v[:12]}")
    assert v == commit["sha"], "head mismatch!"
    print("PUSH OK")


if __name__ == "__main__":
    sys.exit(main())
