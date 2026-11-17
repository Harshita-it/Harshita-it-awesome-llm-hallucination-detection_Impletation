"""Resumable JSON-lines storage. A run may be interrupted (Colab disconnects); we resume by question_id."""
import json
from pathlib import Path


def read_jsonl(path):
    """Read all VALID lines. A truncated last line (interrupted write) is ignored."""
    p = Path(path)
    if not p.exists():
        return []
    out = []
    with open(p) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                break
    return out


def repair_jsonl(path):
    """Rewrite the file with only its valid lines (call once at the start of a run). Returns number of valid lines."""
    p = Path(path)
    if not p.exists():
        return 0
    recs = read_jsonl(p)
    with open(p, "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    return len(recs)


def append_jsonl(path, obj):
    with open(path, "a") as f:
        f.write(json.dumps(obj) + "\n")
        f.flush()


def done_ids(path, key="question_id"):
    return {r[key] for r in read_jsonl(path)}
