#!/usr/bin/env python
"""Freeze the protocol and make the freeze survive Colab session resets.

  python scripts/11_freeze_protocol.py --freeze     records a SHA-256 of config + code, commits them to a local git repo, tags protocol-v1
  python scripts/11_freeze_protocol.py --restore    new session: recreates the tag ONLY IF the code is byte-identical to what was frozen
The record is stored in data/protocol_freeze.json (on Drive in the notebook). Any code change after freezing is refused on purpose:
it would be an undisclosed protocol change. Make a new version (protocol-v1.1) and log it in protocol_changes.md instead."""
import argparse
import json
import subprocess
import time

from _common import ROOT, code_digest, code_files, git_tag_exists

REC = ROOT / "data" / "protocol_freeze.json"


files = code_files
digest = code_digest


def git(*args):
    return subprocess.run(["git", "-C", str(ROOT), "-c", "user.name=protocol-freeze", "-c", "user.email=freeze@localhost", *args],
                          capture_output=True, text=True)


def make_tag():
    if not (ROOT / ".git").exists():
        r = git("init")
        if r.returncode:
            raise SystemExit("git init failed: " + r.stderr)
    (ROOT / ".gitignore").write_text("__pycache__/\n*.pyc\n")
    git("add", ".gitignore", "configs", "src", "scripts")
    r = git("commit", "-m", "protocol-v1 freeze", "--allow-empty")
    if r.returncode:
        raise SystemExit("git commit failed: " + r.stderr + r.stdout)
    r = git("tag", "protocol-v1")
    if r.returncode and not git_tag_exists():
        raise SystemExit("git tag failed: " + r.stderr)
    return git("rev-parse", "HEAD").stdout.strip()


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--freeze", action="store_true")
    g.add_argument("--restore", action="store_true")
    a = ap.parse_args()
    d = digest()
    if a.freeze:
        if REC.exists():
            old = json.load(open(REC))
            if old["sha256"] != d:
                raise SystemExit("Already frozen with DIFFERENT code. Refusing to overwrite. Create protocol-v1.1 and log the change.")
            print("Already frozen with identical code:", old["sha256"][:16]); make_tag(); return
        REC.parent.mkdir(exist_ok=True)
        commit = make_tag()
        json.dump({"sha256": d, "commit": commit, "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                   "files": [str(p.relative_to(ROOT)) for p in files()]}, open(REC, "w"), indent=2)
        print(f"FROZEN. tag protocol-v1 -> {commit[:10]} | code sha256 {d[:16]}... | record {REC}")
    else:
        if not REC.exists():
            raise SystemExit("No freeze record found: the protocol has not been frozen.")
        old = json.load(open(REC))
        if old["sha256"] != d:
            raise SystemExit("Code differs from the frozen version (undisclosed change). Do not run the test part. "
                             "Restore the original files or create protocol-v1.1 with a log entry.")
        make_tag()
        print("Freeze verified: code identical to the frozen version; tag protocol-v1 present:", git_tag_exists())


if __name__ == "__main__":
    main()
