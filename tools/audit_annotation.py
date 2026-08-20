#!/usr/bin/env python3
"""Audit TIM-style annotations for prompt leakage and report-style drift.

The utility is read-only: it never rewrites the annotation. It supports both
TIM's ``{"train": [...], "val": [...], "test": [...]}`` structure and the
``id -> [report]`` JSON files emitted during validation.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

REPORT_FIELDS = (
    "report",
    "context_report",
    "Findings",
    "findings",
)
PROGRESSION_FIELDS = (
    "progressions",
    "progression",
    "Changes",
    "changes",
)
TEXT_FIELDS = REPORT_FIELDS + PROGRESSION_FIELDS

PROMPT_PATTERNS = {
    "user_role": re.compile(r"\buser\s*:", re.IGNORECASE),
    "assistant_role": re.compile(r"\bassistant\s*:", re.IGNORECASE),
    "image_placeholder": re.compile(r"<\s*image\s*>", re.IGNORECASE),
    "progression_placeholder": re.compile(r"<\s*progression\s*>", re.IGNORECASE),
    "radiologist_instruction": re.compile(r"you are a radiologist", re.IGNORECASE),
    "edit_program_instruction": re.compile(r"clinical edit program", re.IGNORECASE),
    "preserve_instruction": re.compile(
        r"preserve clinically stable facts|avoid unnecessary rewriting",
        re.IGNORECASE,
    ),
}

STYLE_PATTERNS = {
    "reply_or_correspondence": re.compile(
        r"\bin reply to\b|\bin response to\b|\bto the radiologist\b",
        re.IGNORECASE,
    ),
    "telephone_or_email": re.compile(
        r"\btelephone\b|\be-?mail\b|\bfax(?:ed)?\b",
        re.IGNORECASE,
    ),
    "administrative": re.compile(
        r"\bpresident\b|\bwhite house\b|\bon behalf of\b|\bdepartment of\b|"
        r"\bthank you for\b|\bunited states of america\b",
        re.IGNORECASE,
    ),
    "non_chest_topic": re.compile(
        r"\bpancreatitis\b|\bcerebral\b|\bbrain\b|\bneurosurgery\b|"
        r"\bliver transplantation\b|\bhemodialysis\b|\bperipancreatic\b",
        re.IGNORECASE,
    ),
}


def _text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return " ".join(_text(item) for item in value)
    if isinstance(value, dict):
        return " ".join(f"{key}: {_text(item)}" for key, item in value.items())
    return str(value)


def _as_splits(payload: Any) -> dict[str, list[dict[str, Any]]]:
    if isinstance(payload, list):
        if not all(isinstance(item, dict) for item in payload):
            raise ValueError("A top-level JSON list must contain objects.")
        return {"records": payload}

    if not isinstance(payload, dict):
        raise ValueError("Expected a JSON object or a list of objects.")

    standard = {
        key: value
        for key, value in payload.items()
        if isinstance(value, list) and (not value or isinstance(value[0], dict))
    }
    if standard and set(standard).intersection({"train", "val", "test"}):
        return standard

    # Validation reference format: {sample_id: [report]}.
    if all(isinstance(value, (str, list)) for value in payload.values()):
        return {
            "flat_reports": [
                {"id": sample_id, "report": _text(value)}
                for sample_id, value in payload.items()
            ]
        }
    raise ValueError("Unsupported annotation JSON structure.")


def _matched(patterns: dict[str, re.Pattern[str]], text: str) -> list[str]:
    return [name for name, pattern in patterns.items() if pattern.search(text)]


def _field_summary(
    records: list[dict[str, Any]],
    field: str,
    max_examples: int,
) -> dict[str, Any]:
    values = [_text(record.get(field)) for record in records]
    present = [(record, value) for record, value in zip(records, values) if value.strip()]
    prompt_counts: Counter[str] = Counter()
    style_counts: Counter[str] = Counter()
    prompt_examples: list[dict[str, str]] = []
    style_examples: list[dict[str, str]] = []
    lengths = []

    for record, value in present:
        lengths.append(len(value.split()))
        prompt_hits = _matched(PROMPT_PATTERNS, value)
        style_hits = _matched(STYLE_PATTERNS, value)
        prompt_counts.update(prompt_hits)
        style_counts.update(style_hits)
        if prompt_hits and len(prompt_examples) < max_examples:
            prompt_examples.append(
                {
                    "id": str(record.get("id", "")),
                    "markers": ", ".join(prompt_hits),
                    "text": value[:300],
                }
            )
        if style_hits and len(style_examples) < max_examples:
            style_examples.append(
                {
                    "id": str(record.get("id", "")),
                    "markers": ", ".join(style_hits),
                    "text": value[:300],
                }
            )

    present_count = len(present)
    prompt_record_count = sum(
        bool(_matched(PROMPT_PATTERNS, value)) for _, value in present
    )
    style_record_count = sum(
        bool(_matched(STYLE_PATTERNS, value)) for _, value in present
    )
    return {
        "records": len(records),
        "present": present_count,
        "missing": len(records) - present_count,
        "mean_tokens": round(statistics.fmean(lengths), 3) if lengths else 0.0,
        "median_tokens": round(statistics.median(lengths), 3) if lengths else 0.0,
        "prompt_contaminated_records": prompt_record_count,
        "prompt_contamination_rate": round(
            prompt_record_count / present_count,
            6,
        ) if present_count else 0.0,
        "prompt_marker_counts": dict(sorted(prompt_counts.items())),
        "style_flagged_records": style_record_count,
        "style_flag_rate": round(style_record_count / present_count, 6) if present_count else 0.0,
        "style_marker_counts": dict(sorted(style_counts.items())),
        "prompt_examples": prompt_examples,
        "style_examples": style_examples,
    }


def audit(payload: Any, max_examples: int = 5) -> dict[str, Any]:
    splits = _as_splits(payload)
    result: dict[str, Any] = {"splits": {}, "totals": {}}
    total_records = 0
    all_ids: list[str] = []
    total_prompt_contamination = 0
    total_text_records = 0

    for split_name, records in splits.items():
        ids = [str(record.get("id", "")) for record in records if record.get("id") is not None]
        all_ids.extend(ids)
        fields = {
            field: _field_summary(records, field, max_examples)
            for field in TEXT_FIELDS
            if any(field in record for record in records)
        }
        split_prompt = sum(
            summary["prompt_contaminated_records"] for summary in fields.values()
        )
        split_text_records = sum(summary["present"] for summary in fields.values())
        total_prompt_contamination += split_prompt
        total_text_records += split_text_records
        total_records += len(records)
        result["splits"][split_name] = {
            "records": len(records),
            "unique_ids": len(set(ids)),
            "duplicate_ids": len(ids) - len(set(ids)),
            "missing_report": sum(not _text(record.get("report")).strip() for record in records),
            "missing_image_path": sum(
                "image_path" in record and not record.get("image_path") for record in records
            ),
            "fields": fields,
        }

    id_counts = Counter(identifier for identifier in all_ids if identifier)
    result["totals"] = {
        "records": total_records,
        "unique_ids": len(id_counts),
        "duplicate_id_occurrences": sum(count - 1 for count in id_counts.values()),
        "prompt_contaminated_text_fields": total_prompt_contamination,
        "text_fields_present": total_text_records,
        "prompt_contamination_rate": round(
            total_prompt_contamination / total_text_records,
            6,
        ) if total_text_records else 0.0,
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("annotation", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--max-examples", type=int, default=5)
    parser.add_argument(
        "--fail-on-contamination",
        action="store_true",
        help="return exit status 2 when prompt-style contamination is found",
    )
    args = parser.parse_args()

    with args.annotation.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    summary = audit(payload, max_examples=args.max_examples)
    rendered = json.dumps(summary, indent=2, ensure_ascii=False)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    if args.fail_on_contamination and summary["totals"]["prompt_contaminated_text_fields"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
