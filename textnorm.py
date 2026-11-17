"""Answer extraction, normalization and labelling.

Normalization follows the OFFICIAL TriviaQA evaluation normalization (as reproduced in the T5 authors' qa_utils.py, which states
that it matches the official script): lowercase; replace every punctuation character, including the characters
(left/right single quote, acute accent, backtick, underscore), by a SPACE; remove the articles a/an/the; collapse whitespace.
NOTE: this differs from the SQuAD normalization, which DELETES punctuation ("U.S.A." -> "usa"). The dataset's
`normalized_aliases` follow the TriviaQA style, so using the SQuAD style would mislabel correct answers as wrong.
Still to do (checklist): compare once with the official script file in the TriviaQA repository itself.
"""
import re
import string

_PUNCT = set(string.punctuation + "\u2018\u2019\u00b4`_")
_ARTICLES = re.compile(r"\b(a|an|the)\b")


def normalize_answer(text: str) -> str:
    text = text.lower()
    text = "".join(" " if ch in _PUNCT else ch for ch in text)
    text = _ARTICLES.sub(" ", text)
    return " ".join(text.split())


def normalize_squad_style(text: str) -> str:
    """Kept ONLY to demonstrate the difference in tests/sensitivity checks. Not used for labels."""
    text = text.lower()
    text = "".join(ch for ch in text if ch not in set(string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())

def extract_short_answer(generated_text: str) -> str:
    """Strip, keep the first line, remove trailing periods (protocol Section 6, step 1)."""
    text = generated_text.strip()
    if not text:
        return ""
    first_line = text.split("\n")[0].strip()
    return first_line.rstrip(".").strip()


def is_correct(generated_text: str, normalized_aliases) -> bool:
    """Exact-match label: True if the normalized short answer equals any normalized alias."""
    answer = normalize_answer(extract_short_answer(generated_text))
    aliases = {normalize_answer(a) for a in normalized_aliases}
    return answer in aliases and answer != ""


def is_correct_containment(generated_text: str, normalized_aliases) -> bool:
    """Sensitivity label: a normalized alias occurs inside the normalized answer (word-boundary match)."""
    answer = normalize_answer(extract_short_answer(generated_text))
    if not answer:
        return False
    padded = f" {answer} "
    return any(f" {normalize_answer(a)} " in padded for a in normalized_aliases if normalize_answer(a))
