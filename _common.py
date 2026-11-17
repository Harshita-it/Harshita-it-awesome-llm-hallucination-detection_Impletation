"""Shared helpers for the pipeline scripts."""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

RAW = ROOT / "results" / "raw"
GEN = RAW / "generations.jsonl"
SEM = RAW / "semantic.jsonl"


def load_cfg():
    return yaml.safe_load(open(ROOT / "configs" / "protocol_v1.yaml"))


def git_commit():
    try:
        return subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10).stdout.strip() or None
    except Exception:  # noqa: BLE001
        return None


def git_tag_exists(tag="protocol-v1"):
    try:
        r = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "-q", "--verify", f"refs/tags/{tag}"], capture_output=True, text=True, timeout=10)
        return r.returncode == 0
    except Exception:  # noqa: BLE001
        return False


def select_rows(df, split):
    """split in pilot|calibration|test|all. Pilot rows first, then by pool_index."""
    if split == "pilot":
        sel = df[df["is_pilot"].astype(bool)]
    elif split == "calibration":
        sel = df[df["split"] == "calibration"]
    elif split == "test":
        sel = df[df["split"] == "test"]
    else:
        sel = df
    return sel.assign(_p=~sel["is_pilot"].astype(bool)).sort_values(["_p", "pool_index"]).drop(columns="_p")


def require_test_permission(split, allowed):
    if split in ("test", "all") and not allowed:
        raise SystemExit("Refusing to touch the TEST part without --allow-test.\n"
                         "Protocol: run the 50-question pilot first (it decides pool size 1000 vs 400 from timing only), "
                         "complete experiment_protocol.md Section 14 and tag protocol-v1.")


def load_subset_with_text(cfg):
    """Rebuild the subset (with question text and aliases) from the saved ids and a fresh download; verify the manifest hash."""
    from uncertainty import data as D
    ids = pd.read_csv(ROOT / "data" / "subset_question_ids.csv", dtype={"question_id": str})
    manifest = json.load(open(ROOT / "data" / "subset_manifest.json"))
    if D.ids_hash(ids) != manifest["ids_sha256"]:
        raise SystemExit("data/subset_question_ids.csv does not match its manifest hash: do not continue.")
    from datasets import load_dataset
    ds = cfg["dataset"]
    raw = load_dataset(ds["hf_name"], ds["config"], split=ds["split"])
    clean, _ = D.dedupe(D.rows_from_records(raw))
    clean, _ = D.drop_unlabellable(clean)
    return D.attach_text(ids, clean).sort_values("pool_index").reset_index(drop=True)


def code_files():
    return sorted([*ROOT.glob("configs/*.yaml"), *ROOT.glob("src/uncertainty/*.py"), *ROOT.glob("scripts/*.py")])


def code_digest():
    h = hashlib.sha256()
    for p in code_files():
        h.update(str(p.relative_to(ROOT)).encode())
        h.update(p.read_bytes())
    return h.hexdigest()


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def freeze_record():
    p = ROOT / "data" / "protocol_freeze.json"
    return json.load(open(p)) if p.exists() else None


def freeze_info():
    rec = freeze_record()
    return {"tag_exists": git_tag_exists(), "record": rec, "code_matches_record": (rec["sha256"] == code_digest()) if rec else None}


def audit_from_disk(mode, cfg, strict_freeze=True):
    """Run the pre-evaluation audit on whatever is on disk (inputs that are missing make their checks SKIPPED, never passed)."""
    from uncertainty.audit import load_audit_inputs, run_audit
    df = pd.read_csv(ROOT / "results" / "per_question.csv")
    return run_audit(df, cfg, mode, freeze=freeze_info() if mode == "final" else None, strict_freeze=strict_freeze,
                     **load_audit_inputs(ROOT))
