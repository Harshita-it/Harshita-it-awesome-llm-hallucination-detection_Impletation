"""Safe diagnostics collection for debugging (Prompt 11). Never prints environment variables; redacts token-like strings."""
import re

_PATTERNS = [
    (re.compile(r"hf_[A-Za-z0-9]{16,}"), "hf_[REDACTED]"),
    (re.compile(r"sk-[A-Za-z0-9_\-]{16,}"), "sk-[REDACTED]"),
    (re.compile(r"(?i)(api[_-]?key|token|secret|password)\s*[=:]\s*\S+"), r"\1=[REDACTED]"),
    (re.compile(r"(?i)authorization:\s*bearer\s+\S+"), "Authorization: Bearer [REDACTED]"),
]


def redact(text: str) -> str:
    for pat, repl in _PATTERNS:
        text = pat.sub(repl, text)
    return text


def tail(path, n=60):
    try:
        with open(path, errors="replace") as f:
            return "".join(f.readlines()[-n:])
    except FileNotFoundError:
        return f"(file not found: {path})\n"
