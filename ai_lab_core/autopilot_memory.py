"""Reconcile durable lessons from the local worker's completion receipts."""
import hashlib
from pathlib import Path

from . import controller, jobs, registry
from .autopilot_runtime import _marker, consumed
from .controller_contract import sha256, validate
from .controller_engine import _METRIC, _json, _relative_gain


SKIP = {"rejected_development", "no_gain_development", "rejected_holdout", "no_gain_holdout"}
_KEEP = 128


def digest(value):
    return hashlib.sha256(value).hexdigest()


def variant_key(variant, identity):
    return digest(_json({"configuration": {k: v for k, v in variant.items() if k != "id"}, "assets": identity}).encode())


def context_key(protocol, source, runtime, baseline_key):
    policy = {k: v for k, v in protocol.items() if k not in ("variants", "baseline_id", "evaluation_scope")}
    return digest(_json({"policy": policy, "data": {k: source[k] for k in ("manifest", "dev", "test")},
                         "runtime": runtime, "baseline": baseline_key}).encode())


def _evidence(job, settings):
    if job["metadata"].get("autonomy") is not True:
        raise registry.RegistryError("job_unverified", "An autonomous local controller job is required.")
    path, raw = registry._receipt(job, "optimization.search", settings)
    summary = controller._report({"action": "report", "report_path": str(path)}, settings)
    report = controller._json(raw)
    request_raw = controller._read(path.with_name("request.json"), controller.MAX_REQUEST_BYTES, owned=True)
    request = controller._json(request_raw)
    provenance = report["provenance"]
    hardware = provenance["hardware"]
    protocol = validate(request["protocol"])
    source = request["source_sha256"]
    autonomy = request["autonomy"]
    runtime = autonomy["runtime_fingerprint"]
    frozen_raw = controller._read(path.with_name("selection.json"), controller.MAX_REPORT_BYTES, owned=True)
    frozen = controller._json(frozen_raw)
    if (summary.get("report_sha256") != digest(raw)
            or digest(request_raw) != job["metadata"]["request_sha256"]
            or digest(request_raw) != hardware["request_sha256"]
            or request["workspace"] != str(Path(settings["workspace_root"]).resolve())
            or request["output_dir"] != str(path.parent) or autonomy["profile_state"] != settings["state_dir"]
            or not sha256(runtime) or runtime != hardware["autonomy_runtime_sha256"]
            or hardware.get("metal_available") is not True or "apple-silicon-mlx" not in hardware["profiles"]
            or digest(_json(protocol).encode()) != provenance["protocol_sha256"]
            or source["protocol"] != hardware["protocol_source_sha256"]
            or {k: source[k] for k in ("manifest", "dev", "test")} != hardware["source_data_sha256"]
            or source["manifest"] != provenance["dataset_manifest_sha256"]
            or {f"{k}.jsonl": source[k] for k in ("dev", "test")} != provenance["dataset_sha256"]
            or protocol["baseline_id"] != summary["baseline_variant_id"]
            or protocol.get("evaluation_scope", "full") != summary.get("evaluation_scope", "full")
            or digest(frozen_raw) != summary["selection_sha256"]
            or any(frozen.get(k) != report.get(k) for k in (
                "kind", "baseline_variant_id", "selected_variant_id", "selection_basis", "evaluation_scope", "provenance", "dev",
            ))):
        raise registry.RegistryError("integrity_error", "The campaign evidence differs from the local job receipts.")
    if summary["test"]:
        receipt = _marker(settings["state_dir"], source["test"])
        if (not consumed(settings["state_dir"], source["test"])
                or controller._json(controller._read(receipt, 512, owned=True)) != {
                    "test_sha256": source["test"], "request_sha256": digest(request_raw),
                }):
            raise registry.RegistryError("integrity_error", "The final test lacks its autonomous consumption receipt.")
    return protocol, source, runtime, summary, provenance["identities"], report.get("comparison", {})


def _lesson(job, evidence):
    protocol, source, runtime, summary, identities, comparison = evidence
    variants = {variant["id"]: variant for variant in protocol["variants"]}
    if set(variants) != set(identities) or set(variants) != set(summary["dev"]):
        raise ValueError("evidence variants")
    keys = {k: variant_key(v, registry._identity(identities[k])) for k, v in variants.items()}
    baseline = protocol["baseline_id"]
    reference = summary["dev"][baseline]
    candidates = []
    for identifier, observed in summary["dev"].items():
        if identifier == baseline or not observed["complete"] or not reference["complete"]:
            continue
        final = summary["test"].get(identifier)
        split = "development"
        metrics = observed
        gain = _relative_gain(protocol["objective"], observed, reference)
        if summary["accepted_variant_id"] == identifier:
            split, metrics = "holdout", final
            outcome = "qualified"
        elif not observed["eligible"]:
            outcome = "rejected_development"
        elif final is not None and final["complete"]:
            split, metrics = "holdout", final
            outcome = ("no_gain_holdout" if final["eligible"] and comparison.get("quality_noninferior") is not False
                       else "rejected_holdout")
        elif reference["eligible"] and (gain <= 0 or gain < protocol["minimum_relative_gain"]):
            outcome = "no_gain_development"
        else:
            outcome = "promising_development"
        if split == "holdout":
            gain = _relative_gain(protocol["objective"], final, summary["test"][baseline])
        candidates.append({"variant_id": identifier, "variant_key": keys[identifier], "outcome": outcome,
                           "split": split, "failed_gates": metrics["failed_gates"], "relative_gain": gain,
                           "objective_value": metrics[_METRIC[protocol["objective"]]], "quality": metrics["quality"]})
    return {"job_id": job["id"], "status": summary["status"], "objective": protocol["objective"],
            "context_sha256": context_key(protocol, source, runtime, keys[baseline]), "candidates": candidates,
            "report_sha256": summary["report_sha256"], "selection_sha256": summary["selection_sha256"],
            "dataset_manifest_sha256": source["manifest"], "runtime_sha256": runtime}


def _table(db):
    db.execute("""CREATE TABLE IF NOT EXISTS autonomous_lessons (
        job_id TEXT PRIMARY KEY, finished REAL NOT NULL, lesson TEXT NOT NULL)""")


def _save(db, job, lesson):
    db.execute("INSERT OR IGNORE INTO autonomous_lessons VALUES(?,?,?)", (job["id"], job["finished"], _json(lesson)))
    db.execute("""DELETE FROM autonomous_lessons WHERE job_id NOT IN (
        SELECT job_id FROM autonomous_lessons ORDER BY finished DESC, job_id DESC LIMIT ?)""", (_KEEP,))


def reconcile(settings, limit=100):
    root = Path(settings["state_dir"])
    if not (root / "jobs.sqlite3").is_file():
        return []
    root = jobs._root(root)
    with jobs._db(root) as db:
        _table(db)
        pending = db.execute("""SELECT * FROM jobs WHERE state='succeeded'
            AND json_extract(plan,'$.operation')='optimization.search'
            AND json_extract(plan,'$.metadata.autonomy')=1
            ORDER BY finished DESC, id DESC LIMIT ?""", (_KEEP,)).fetchall()
        recorded = {r["job_id"] for r in db.execute("SELECT job_id FROM autonomous_lessons")}
    verified = []
    for row in pending:
        job = jobs._public(row)
        try:
            lesson = _lesson(job, _evidence(job, settings))
        except (KeyError, OSError, TypeError, ValueError, RecursionError):
            if row["id"] in recorded:
                with jobs._db(root) as db:
                    db.execute("DELETE FROM autonomous_lessons WHERE job_id=?", (row["id"],))
            continue
        if row["id"] not in recorded:
            with jobs._db(root) as db:
                _save(db, job, lesson)
        verified.append(lesson)
    return verified[:limit]


def report(job_id, settings):
    job = registry._job(job_id, settings)
    if job["metadata"].get("autonomy") is not True:
        raise registry.RegistryError("job_unverified", "An autonomous local controller job is required.")
    if job["state"] != "succeeded":
        return {"ok": True, "status": "autopilot_" + job["state"], "job": job, "lesson_recorded": False}
    evidence = _evidence(job, settings)
    lesson = _lesson(job, evidence)
    with jobs._db(jobs._root(settings["state_dir"])) as db:
        _table(db)
        _save(db, job, lesson)
    return {"ok": True, "status": lesson["status"], "job": job, "report": evidence[3],
            "lesson_recorded": True, "lesson": lesson}
