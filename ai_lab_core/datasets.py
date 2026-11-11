"""Bounded JSONL dataset validation without exposing training examples."""
from __future__ import annotations

import hashlib
import json
from .endpoint import failure
from .integration_config import permitted_path

MAX_BYTES = 128 * 1024 * 1024


def validate_dataset(path, settings):
    try:
        source = permitted_path(path, settings)
        files = [source] if source.is_file() else [source / n for n in ("train.jsonl", "valid.jsonl", "test.jsonl") if (source / n).is_file()]
        if not files or any(p.suffix != ".jsonl" or p.stat().st_size > MAX_BYTES for p in files):
            return failure("invalid_dataset", "Use JSONL files no larger than 128 MiB per split.")
        splits, known, overlap = [], {}, 0
        for file in files:
            permitted_path(str(file), settings)
            digest, count, duplicates, invalid = hashlib.sha256(), 0, 0, []
            seen = set()
            with file.open("rb") as stream:
                for lineno, raw in enumerate(stream, 1):
                    digest.update(raw)
                    if len(raw) > 1024 * 1024 or lineno > 100000:
                        return failure("dataset_limit", "Dataset exceeds the row or line size limit.")
                    if not raw.strip():
                        continue
                    try:
                        value = json.loads(raw)
                        if not _valid_row(value):
                            raise ValueError
                    except (ValueError, UnicodeError, RecursionError):
                        if len(invalid) < 20:
                            invalid.append(lineno)
                        continue
                    count += 1
                    row_id = hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
                    duplicates += int(row_id in seen)
                    if row_id in known and known[row_id] != file.name:
                        overlap += 1
                    seen.add(row_id)
                    known[row_id] = file.name
            splits.append({"file": str(file), "rows": count, "sha256": digest.hexdigest(), "duplicate_rows": duplicates,
                           "invalid_lines": invalid, "valid": count > 0 and not invalid})
        valid = all(s["valid"] for s in splits) and overlap == 0
        result = {"ok": valid, "splits": splits, "cross_split_duplicate_rows": overlap,
                  "verification": "format_and_exact_duplicate_check", "quality_verified": False}
        if not valid:
            result["error"] = {"code": "invalid_dataset", "message": "Dataset has invalid or empty splits, or exact overlap between splits."}
        return result
    except (ValueError, OSError, TypeError):
        return failure("invalid_path", "Dataset must be readable inside configured roots.")


def _valid_row(value):
    if not isinstance(value, dict):
        return False
    if isinstance(value.get("text"), str) and value["text"].strip():
        return True
    if isinstance(value.get("prompt"), str) and isinstance(value.get("completion"), str):
        return bool(value["prompt"].strip() and value["completion"].strip())
    messages = value.get("messages")
    return isinstance(messages, list) and bool(messages) and all(
        isinstance(m, dict) and m.get("role") in {"system", "user", "assistant", "tool"}
        and (isinstance(m.get("content"), str) or (m.get("role") == "assistant" and isinstance(m.get("tool_calls"), list)))
        for m in messages)
