"""Versioned project-authored pairs, not authoritative religious judgments."""

import json
from pathlib import Path

from .corpus import load_verses

THEMES = {"commitments", "entrusted_access", "privacy", "uncertainty", "permitted_work"}
FIELDS = {"id", "family", "theme", "task", "positive", "negative", "refs", "interpretation_notes",
          "language", "split", "review_status", "authorship"}


def load_behavior_data(path, corpus_path, theme=None):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if set(data) != {"version", "examples"} or data["version"] != "1.0":
        raise ValueError("Unknown behavioral dataset version/schema")
    canonical = {v.ref for v in load_verses(corpus_path)}
    families, ids = {}, set()
    for row in data["examples"]:
        if set(row) != FIELDS or row["id"] in ids:
            raise ValueError("Invalid behavioral example schema/id")
        ids.add(row["id"])
        if row["split"] not in {"train", "dev", "test"} or row["theme"] not in THEMES:
            raise ValueError("Invalid split/theme")
        if families.setdefault(row["family"], row["split"]) != row["split"]:
            raise ValueError("Scenario family leaks across splits")
        if not row["refs"] or not set(row["refs"]) <= canonical:
            raise ValueError("Verse reference absent from corpus")
        if row["authorship"] != "project-authored" or row["review_status"] not in {"unreviewed", "project-reviewed", "expert-reviewed"}:
            raise ValueError("Invalid authorship/review status")
        for key in ("family", "task", "positive", "negative", "interpretation_notes", "language"):
            if not isinstance(row[key], str) or not row[key].strip():
                raise ValueError(f"Missing behavioral field {key}")
    if theme is not None and theme not in THEMES:
        raise ValueError("Unknown optional behavioral theme")
    examples = [r for r in data["examples"] if theme is None or r["theme"] == theme]
    if {r["split"] for r in examples} != {"train", "dev", "test"}:
        raise ValueError("Need examples in every separate split")
    return examples


def paired_texts(examples):
    # Same task prefix and vocabulary within each pair; responses are approximately
    # length-matched. This reduces, but does not eliminate, style confounding.
    return tuple([f"Task: {r['task']}\nResponse: {r[key]}" for r in examples]
                 for key in ("positive", "negative"))
