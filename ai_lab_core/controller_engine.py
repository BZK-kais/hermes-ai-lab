"""Run a finite local search and a frozen baseline/candidate comparison."""

from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import stat
import tempfile
from time import monotonic

from . import controller_contract as contract
from .endpoint import failure

_FILES = ("dev.jsonl", "test.jsonl")
_MAX_LINE_BYTES = contract.MAX_INPUT_BYTES + contract.MAX_EXPECTED_BYTES + 512
_MAX_DATASET_BYTES = contract.MAX_CASES * _MAX_LINE_BYTES
_MESSAGES = {
    "incomplete": "The controller did not finish.",
    "invalid_request": "The controller protocol, dataset or output directory is invalid.",
    "integrity_error": "A frozen controller input changed during the campaign.",
    "budget_exhausted": "The controller could not complete within its declared budget.",
    "measurement_error": "The controller did not obtain complete valid measurements.",
    "cancelled": "The controller was cancelled.",
    "holdout_consumed": "This holdout was already consumed by an autonomous campaign in this profile.",
}
_METRIC = {
    "quality": "quality",
    "latency": "p95_latency_ms",
    "memory": "mlx_active_plus_cache_bound_bytes",
    "throughput": "requests_per_second",
}
_HIGHER_IS_BETTER = ("quality", "throughput")
_OUTPUT_FLAGS = ("output_parseable", "output_schema_matches")


class _Stop(Exception):
    pass


def _digest(raw):
    return hashlib.sha256(raw).hexdigest()


def _json(value):
    return json.dumps(value, ensure_ascii=True, sort_keys=True, allow_nan=False, separators=(",", ":"))


def _atomic_write(path, value):
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, encoding="utf-8", delete=False) as stream:
        temporary = Path(stream.name)
        try:
            stream.write(_json(value) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)


def _read_file(path, limit):
    flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK
    with os.fdopen(os.open(path, flags), "rb") as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_size > limit:
            raise ValueError
        raw = stream.read(limit + 1)
    if len(raw) > limit:
        raise ValueError
    return raw


def _snapshot(directory):
    manifest = _read_file(directory / "manifest.sha256", 512)
    hashes = {}
    for line in manifest.decode("ascii").splitlines():
        match = re.fullmatch(r"([a-f0-9]{64}) [ *](dev\.jsonl|test\.jsonl)", line)
        if match is None or match[2] in hashes:
            raise ValueError
        hashes[match[2]] = match[1]
    if set(hashes) != set(_FILES):
        raise ValueError
    raw = {filename: _read_file(directory / filename, _MAX_DATASET_BYTES) for filename in _FILES}
    if any(_digest(raw[filename]) != hashes[filename] for filename in _FILES):
        raise ValueError
    provenance = {"dataset_manifest_sha256": _digest(manifest), "dataset_sha256": hashes}
    return raw, provenance


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError
        result[key] = value
    return result


def _reject_constant(_):
    raise ValueError


def _case_count(raw):
    # Count rows without decoding the holdout or inspecting any of its fields.
    count = raw.count(b"\n") + bool(raw and not raw.endswith(b"\n"))
    if not 1 <= count <= contract.MAX_CASES:
        raise ValueError
    return count


def _parse_cases(raw, grader, required_slices):
    _case_count(raw)
    cases = []
    ids = set()
    for line in raw.removesuffix(b"\n").split(b"\n"):
        if len(line) > _MAX_LINE_BYTES:
            raise ValueError
        row = json.loads(
            line.decode("utf-8"), object_pairs_hook=_unique_object, parse_constant=_reject_constant,
        )
        if (not contract.fields(row, ("id", "family", "slice", "input", "expected"))
                or not all(contract.name(row[key]) for key in ("id", "family", "slice"))
                or row["id"] in ids or not contract.text(row["input"], contract.MAX_INPUT_BYTES)):
            raise ValueError
        expected = row["expected"]
        if grader == "exact_text" and not contract.text(expected, contract.MAX_EXPECTED_BYTES, empty=True):
            raise ValueError
        if len(_json(expected)) > contract.MAX_EXPECTED_BYTES:
            raise ValueError
        ids.add(row["id"])
        cases.append(row)
    if len(cases) != _case_count(raw) or not required_slices.keys() <= {row["slice"] for row in cases}:
        raise ValueError
    return cases


def _identify(variant):
    from .skill_worker import adapter_fingerprint, model_fingerprint

    model = model_fingerprint(Path(variant["model"]))
    adapter = adapter_fingerprint(Path(variant["adapter"])) if "adapter" in variant else None
    return {"model_sha256": model, "adapter_sha256": adapter}


def _identity(result, variant):
    if (type(result) is not dict or not contract.sha256(result.get("model_sha256"))
            or "adapter_sha256" not in result):
        raise ValueError
    adapter = result["adapter_sha256"]
    if not (contract.sha256(adapter) if "adapter" in variant else adapter is None):
        raise ValueError
    return {"model_sha256": result["model_sha256"], "adapter_sha256": adapter}


def _p95(values):
    # Nearest rank gives a conservative tail estimate for small fixed splits.
    return sorted(values)[math.ceil(0.95 * len(values)) - 1]


def _measurement(result, variant, cases, frozen_identity):
    records = result.get("records")
    if _identity(result, variant) != frozen_identity:
        raise _Stop("integrity_error")
    if type(records) is not list or len(records) != len(cases):
        raise ValueError
    safe_records = []
    max_tokens = variant["max_tokens"]
    constrained = "output_constraint" in variant
    constraint_flags = ("output_constraint_applied", "constraint_verified") if constrained else ()
    for record, case in zip(records, cases):
        if (type(record) is not dict or record.get("case_id") != case["id"]
                or type(record.get("correct")) is not bool
                or not contract.number(record.get("duration_ms"), 0.000001, 3_600_000)
                or not contract.number(record.get("ttft_ms"), 0, record["duration_ms"])
                or not contract.number(record.get("generated_tokens"), 1, max_tokens, integer=True)):
            raise ValueError
        optional = {
            "output_sha256": contract.sha256,
            **dict.fromkeys(_OUTPUT_FLAGS, lambda value: type(value) is bool),
            "prompt_tokens": lambda value: contract.number(value, 1, 2_048, integer=True),
            "finish_reason": lambda value: value in ("length", "stop"),
            "prefill_has_effect": lambda value: type(value) is bool
            and type(record.get("prompt_tokens")) is int
            and value == (record["prompt_tokens"] > variant["prefill_step_size"] + 1),
        }
        if any(key in record and not valid(record[key]) for key, valid in optional.items()):
            raise ValueError
        if not all(record.get(key) is True for key in constraint_flags):
            raise ValueError
        safe_records.append({
            "case_id": case["id"], "slice": case["slice"],
            **{key: record[key] for key in ("correct", "duration_ms", "ttft_ms", "generated_tokens")},
            **{key: record[key] for key in (*optional, *constraint_flags) if key in record},
        })

    memory_keys = ("peak_mlx_active_bytes", "peak_mlx_cache_sampled_bytes", "peak_process_rss_bytes")
    if not all(contract.number(result.get(key), 0, 2**60, integer=True) for key in memory_keys):
        raise ValueError
    if not contract.number(result.get("load_ms"), 0, 3_600_000):
        raise ValueError
    if constrained and not contract.number(result.get("decoding_setup_ms"), 0, 3_600_000):
        raise ValueError
    generated = sum(record["generated_tokens"] for record in safe_records)
    used = result.get("used_generated_tokens")
    if not contract.number(used, generated + 1, generated + max_tokens, integer=True):
        raise ValueError

    slices = {}
    for record in safe_records:
        slices.setdefault(record["slice"], []).append(record["correct"])
    duration_seconds = sum(record["duration_ms"] for record in safe_records) / 1_000
    return {
        "records": safe_records,
        "case_count": len(safe_records),
        "quality": sum(record["correct"] for record in safe_records) / len(safe_records),
        **{
            f"{key}_fraction": sum(record[key] for record in safe_records) / len(safe_records)
            for key in (*_OUTPUT_FLAGS, "constraint_verified") if all(key in record for record in safe_records)
        },
        "slice_quality": {key: sum(values) / len(values) for key, values in slices.items()},
        "slice_case_counts": {key: len(values) for key, values in slices.items()},
        "p95_latency_ms": _p95([record["duration_ms"] for record in safe_records]),
        "p95_ttft_ms": _p95([record["ttft_ms"] for record in safe_records]),
        "requests_per_second": len(safe_records) / duration_seconds,
        "generated_tokens_per_second": generated / duration_seconds,
        "mlx_active_plus_cache_bound_bytes": result["peak_mlx_active_bytes"]
        + result["peak_mlx_cache_sampled_bytes"],
        **{key: result[key] for key in (*memory_keys, "load_ms", "used_generated_tokens")},
        **({"decoding_setup_ms": result["decoding_setup_ms"]} if constrained else {}),
    }


def _aggregate(trials, constraints, repetitions):
    complete = len(trials) == repetitions and all(trial["ok"] for trial in trials)
    result = {"complete": complete, "eligible": False, "repetitions": [
        {key: value for key, value in trial.items() if key != "records"} for trial in trials
    ]}
    if not complete:
        result["failed_gates"] = ["incomplete_measurement"]
        return result

    result.update(
        quality=min(trial["quality"] for trial in trials),
        slice_quality={
            key: min(trial["slice_quality"][key] for trial in trials)
            for key in trials[0]["slice_quality"]
        },
        p95_latency_ms=max(trial["p95_latency_ms"] for trial in trials),
        mlx_active_plus_cache_bound_bytes=max(trial["mlx_active_plus_cache_bound_bytes"] for trial in trials),
        requests_per_second=min(trial["requests_per_second"] for trial in trials),
    )
    for flag in (*_OUTPUT_FLAGS, "constraint_verified"):
        key = f"{flag}_fraction"
        if all(key in trial for trial in trials):
            result[key] = min(trial[key] for trial in trials)
    failed = []
    if result["quality"] < constraints["quality_min"]:
        failed.append("quality")
    failed.extend(
        "slice:" + key for key, threshold in constraints["minimum_slice_quality"].items()
        if result["slice_quality"][key] < threshold
    )
    if result["p95_latency_ms"] > constraints["max_p95_latency_ms"]:
        failed.append("latency")
    if result["mlx_active_plus_cache_bound_bytes"] > constraints["max_mlx_memory_bytes"]:
        failed.append("memory")
    result.update(eligible=not failed, failed_gates=failed)
    return result


def _relative_gain(objective, candidate, baseline):
    metric = _METRIC[objective]
    value, reference = candidate[metric], baseline[metric]
    improvement = value - reference if objective in _HIGHER_IS_BETTER else reference - value
    # A zero baseline has no finite relative change. Use the positive
    # metric gain on its native scale, which for quality is a fraction.
    return improvement / reference if reference else improvement


class _Campaign:
    def __init__(self, protocol, output, measure, hardware, progress, identify, verify, start, deadline):
        self.protocol = protocol
        self.output = output
        self.measure = measure
        self.progress = progress
        self.identify = identify
        self.verify = verify
        self.start = start
        self.deadline = start + protocol["budget"]["max_wall_seconds"]
        if deadline is not None:
            self.deadline = min(self.deadline, deadline)
        self.variants = {variant["id"]: variant for variant in protocol["variants"]}
        self.baseline_id = protocol["baseline_id"]
        self.holdout_candidate_id = protocol.get("holdout_candidate_id")
        self.development_only = protocol.get("evaluation_scope") == "development"
        self.scope_field = {key: protocol[key] for key in ("evaluation_scope",) if key in protocol}
        self.selection_basis = "predeclared_pair" if self.holdout_candidate_id else "development"
        pair = {"holdout_candidate_id": self.holdout_candidate_id} if self.holdout_candidate_id else {}
        self.sources = [__file__, contract.__file__]
        if any("output_constraint" in variant for variant in self.variants.values()):
            from . import json_constraints

            self.sources.append(json_constraints.__file__)
        self.repetitions = protocol["repetitions"]
        self.constraints = protocol["constraints"]
        self.objective = protocol["objective"]
        self.report = {
            "kind": contract.VERSION,
            "ok": False,
            "status": "incomplete",
            "baseline_variant_id": self.baseline_id,
            "selected_variant_id": self.holdout_candidate_id,
            "selection_basis": self.selection_basis,
            **pair,
            **self.scope_field,
            "accepted_variant_id": None,
            "optimization_gain_verified": False,
            "activation_performed": False,
            "objective": self.objective,
            "constraints": self.constraints,
            "budget": self.protocol["budget"],
            "concurrency": 1,
            "generated_tokens_accounted": 0,
            "dev": {},
            "test": {},
            "provenance": {
                **pair,
                "protocol_sha256": _digest(_json(protocol).encode()),
                "source_sha256": {
                    Path(path).name: _digest(Path(path).read_bytes())
                    for path in self.sources
                },
                "hardware": deepcopy(hardware),
            },
        }

    def checkpoint(self, stage=None, **details):
        if monotonic() >= self.deadline:
            raise _Stop("budget_exhausted")
        if stage is not None and self.progress is not None:
            try:
                self.progress({"stage": stage, "elapsed_seconds": monotonic() - self.start, **details})
            except Exception:
                raise _Stop("cancelled") from None
            if monotonic() >= self.deadline:
                raise _Stop("budget_exhausted")

    def identities(self, error):
        result = {}
        for identifier, variant in self.variants.items():
            self.checkpoint()
            try:
                result[identifier] = _identity(self.identify(deepcopy(variant)), variant)
            except Exception:
                raise _Stop(error) from None
            self.checkpoint()
        return result

    def verify_sources(self):
        if self.verify is not None:
            try:
                self.verify()
            except Exception:
                raise _Stop("integrity_error") from None
        self.checkpoint()

    def check_freeze(self, path, digest, length):
        try:
            if _digest(_read_file(path, length)) != digest:
                raise ValueError
        except (OSError, ValueError):
            raise _Stop("integrity_error") from None

    def check_integrity(self, directory, snapshot_provenance, identities):
        self.checkpoint()
        try:
            _, observed = _snapshot(directory)
            sources = self.report["provenance"]["source_sha256"]
            if observed != snapshot_provenance or any(
                _digest(Path(path).read_bytes()) != sources[Path(path).name]
                for path in self.sources
            ):
                raise ValueError
        except (OSError, ValueError):
            raise _Stop("integrity_error") from None
        if self.identities("integrity_error") != identities:
            raise _Stop("integrity_error")
        self.verify_sources()

    def preflight_budget(self, dev_count, test_count):
        tokens_per_round = (dev_count + 1) * sum(v["max_tokens"] for v in self.variants.values())
        if test_count is not None:
            baseline_cap = self.variants[self.baseline_id]["max_tokens"]
            contender_cap = max(
                (v["max_tokens"] for identifier, v in self.variants.items() if identifier != self.baseline_id),
                default=0,
            )
            tokens_per_round += (test_count + 1) * (baseline_cap + contender_cap)
        ceiling = self.repetitions * tokens_per_round
        self.report["generated_token_preflight_ceiling"] = ceiling
        if ceiling > self.protocol["budget"]["max_generated_tokens"]:
            raise _Stop("budget_exhausted")

    def trial(self, identifier, cases, repetition, split):
        variant = self.variants[identifier]
        self.checkpoint("trial", variant_id=identifier, split=split, repetition=repetition)
        ceiling = (len(cases) + 1) * variant["max_tokens"]
        used = self.report["generated_tokens_accounted"]
        if used + ceiling > self.protocol["budget"]["max_generated_tokens"]:
            raise _Stop("budget_exhausted")
        # Reserve the full amount first. Incomplete/invalid trials cannot
        # substantiate actual consumption, so their full reservation remains.
        self.report["generated_tokens_accounted"] = used + ceiling
        record = {"variant_id": identifier, "split": split, "repetition": repetition, "ok": False}
        failure_code = "measurement_error"
        options = deepcopy({**variant, "grader": self.protocol["grader"]})
        try:
            result = self.measure(options, deepcopy(cases), repetition, split, self.deadline)
            if type(result) is not dict or result.get("ok") is not True:
                error = result.get("error") if type(result) is dict else None
                code = error.get("code") if type(error) is dict else None
                if code in ("budget_exhausted", "integrity_error", "cancelled"):
                    raise _Stop(code)
                if code in (
                    "invalid_prompt_bytes", "invalid_chat_template", "invalid_prompt_tokens",
                    "invalid_generation_stream", "invalid_output_constraint",
                ):
                    failure_code = code
                raise ValueError
            reported_tokens = result.get("used_generated_tokens")
            if type(reported_tokens) is int and reported_tokens > ceiling:
                raise _Stop("budget_exhausted")
            measured = _measurement(result, variant, cases, self.report["provenance"]["identities"][identifier])
        except _Stop as exc:
            record["error"] = {"code": exc.args[0]}
            record["terminal"] = True
        except Exception:
            record["error"] = {"code": failure_code}
        else:
            self.report["generated_tokens_accounted"] = used + measured["used_generated_tokens"]
            record.update(ok=True, **measured)
        return record

    def split(self, name, identifiers, cases, stream):
        traces = {identifier: [] for identifier in identifiers}
        for repetition in range(self.repetitions):
            self.checkpoint("round", split=name, repetition=repetition)
            ordered_cases = list(cases)
            seed = _digest(f"{contract.VERSION}:{name}:{repetition}".encode())
            random.Random(int(seed, 16)).shuffle(ordered_cases)
            offset = repetition % len(identifiers)
            for identifier in identifiers[offset:] + identifiers[:offset]:
                if traces[identifier] and not traces[identifier][-1]["ok"]:
                    continue
                trial = self.trial(identifier, ordered_cases, repetition, name)
                traces[identifier].append(trial)
                stream.write(_json(trial) + "\n")
                stream.flush()
                self.report[name][identifier] = _aggregate(
                    traces[identifier], self.constraints, self.repetitions,
                )
                if trial.get("terminal"):
                    raise _Stop(trial["error"]["code"])
                self.checkpoint()
        return self.report[name]

    def choose(self):
        eligible = [identifier for identifier, result in self.report["dev"].items() if result["eligible"]]
        if not eligible:
            return None

        def rank(identifier):
            metrics = self.report["dev"][identifier]
            value = metrics[_METRIC[self.objective]]
            primary = -value if self.objective in _HIGHER_IS_BETTER else value
            return primary, -metrics["quality"], metrics["p95_latency_ms"], list(self.variants).index(identifier)

        return min(eligible, key=rank)

    def decide(self, selected):
        baseline = self.report["test"][self.baseline_id]
        candidate = self.report["test"][selected]
        if not baseline["complete"]:
            raise _Stop("measurement_error")
        baseline_eligible = baseline["eligible"]
        candidate_eligible = candidate["eligible"]
        if self.holdout_candidate_id:
            baseline_eligible = baseline_eligible and self.report["dev"][self.baseline_id]["eligible"]
            candidate_eligible = candidate_eligible and self.report["dev"][selected]["eligible"]
        if selected != self.baseline_id and candidate["complete"]:
            gain = _relative_gain(self.objective, candidate, baseline)
            quality_difference = min(
                measured["quality"] - reference["quality"]
                for measured, reference in zip(candidate["repetitions"], baseline["repetitions"])
            )
            noninferior = quality_difference >= -self.protocol["quality_noninferiority_tolerance"] - 1e-12
            gain_verified = gain > 0 and gain >= self.protocol["minimum_relative_gain"]
            self.report["comparison"] = {
                "objective_relative_gain": gain,
                "minimum_paired_quality_difference": quality_difference,
                "quality_noninferior": noninferior,
                "objective_gain_verified": gain_verified,
            }
            if candidate_eligible and noninferior and (gain_verified or not baseline["eligible"]):
                self.report.update(
                    accepted_variant_id=selected, optimization_gain_verified=gain_verified,
                )
                return "qualified_candidate"
        if baseline_eligible:
            self.report["accepted_variant_id"] = self.baseline_id
            return "retained_baseline"
        return "no_qualified_configuration"

    def execute(self, directory, stream, before_holdout=None):
        self.checkpoint("preflight")
        raw, snapshot_provenance = _snapshot(directory)
        self.report["provenance"].update(snapshot_provenance)
        grader = self.protocol["grader"]["kind"]
        slices = self.constraints["minimum_slice_quality"]
        dev = _parse_cases(raw["dev.jsonl"], grader, slices)
        test_count = None if self.development_only else _case_count(raw["test.jsonl"])
        self.preflight_budget(len(dev), test_count)
        identities = self.identities("invalid_request")
        self.report["provenance"]["identities"] = identities
        self.verify_sources()
        self.report["provenance"]["case_order"] = "controller-v1 split/repetition seeded shuffle"
        _atomic_write(self.output / "report.json", {**self.report, **_failed("incomplete")})
        self.split("dev", list(self.variants), dev, stream)
        if self.development_only and not all(
            self.report["dev"][identifier]["complete"] for identifier in self.variants
        ):
            raise _Stop("measurement_error")
        selected = self.holdout_candidate_id or self.choose()
        if self.holdout_candidate_id and not all(
            self.report["dev"][identifier]["complete"] for identifier in (self.baseline_id, selected)
        ):
            raise _Stop("measurement_error")
        if selected is None and not any(result["complete"] for result in self.report["dev"].values()):
            raise _Stop("measurement_error")
        freeze = {
            "kind": contract.VERSION,
            "selected_variant_id": selected,
            "selection_basis": self.selection_basis,
            **({"holdout_candidate_id": selected} if self.holdout_candidate_id else {}),
            "baseline_variant_id": self.baseline_id,
            **self.scope_field,
            "provenance": self.report["provenance"],
            "dev": self.report["dev"],
        }
        freeze_path = self.output / "selection.json"
        _atomic_write(freeze_path, freeze)
        freeze_bytes = (_json(freeze) + "\n").encode()
        freeze_digest = _digest(freeze_bytes)
        self.report.update(selected_variant_id=selected, selection_sha256=freeze_digest)
        self.checkpoint("selection_frozen", variant_id=selected)
        self.check_integrity(directory, snapshot_provenance, identities)
        self.check_freeze(freeze_path, freeze_digest, len(freeze_bytes))
        if self.development_only:
            self.checkpoint()
            return "development_complete"
        if selected is None:
            return "no_qualified_configuration"

        if before_holdout is not None:
            # A technical failure in development must not spend the holdout.
            if not all(self.report["dev"][identifier]["complete"] for identifier in self.variants):
                raise _Stop("measurement_error")
            self.checkpoint()
            try:
                before_holdout()
            except FileExistsError:
                raise _Stop("holdout_consumed") from None
            except (OSError, ValueError):
                raise _Stop("integrity_error") from None
        test = _parse_cases(raw["test.jsonl"], grader, slices)
        for key in ("id", "family"):
            if {row[key] for row in dev} & {row[key] for row in test}:
                raise ValueError
        finalists = list(dict.fromkeys((self.baseline_id, selected)))
        self.split("test", finalists, test, stream)
        self.check_integrity(directory, snapshot_provenance, identities)
        self.check_freeze(freeze_path, freeze_digest, len(freeze_bytes))
        self.checkpoint()
        return self.decide(selected)


def run(protocol, dataset_dir, output_dir, measure, hardware, progress=None, *,
        identify=None, verify=None, deadline=None, before_holdout=None):
    """Run in a fresh output directory. Callbacks must honor the monotonic deadline."""
    start = monotonic()
    try:
        protocol = contract.validate(protocol)
        dataset = Path(dataset_dir)
        output = Path(output_dir)
        if (type(hardware) is not dict or not output.is_dir() or output.is_symlink()
                or any(os.path.lexists(output / name) for name in ("report.json", "selection.json", "trials.jsonl"))):
            raise ValueError
        # Validate serialization before using hardware in any artifact.
        if len(_json(hardware)) > 32_768 or (
            deadline is not None and not contract.number(deadline, 0, 2**60)
        ):
            raise ValueError
        campaign = _Campaign(protocol, output, measure, hardware, progress, identify or _identify, verify, start, deadline)
        report_path = output / "report.json"
        trace_fd = os.open(output / "trials.jsonl", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except (OSError, TypeError, ValueError, RecursionError):
        return _failed("invalid_request")

    report = campaign.report
    with os.fdopen(trace_fd, "w", encoding="utf-8") as stream:
        try:
            _atomic_write(report_path, {**report, **_failed("incomplete")})
            status = campaign.execute(dataset, stream, before_holdout)
        except _Stop as exc:
            report.update(_failed(exc.args[0]))
        except (OSError, TypeError, ValueError, RecursionError):
            report.update(_failed("invalid_request"))
        else:
            report.update(ok=True, status=status)
    report["elapsed_seconds"] = round(monotonic() - start, 6)
    try:
        _atomic_write(report_path, report)
    except OSError:
        return _failed("measurement_error")
    return report


def _failed(code):
    return {
        "kind": contract.VERSION, "status": code,
        "accepted_variant_id": None, "activation_performed": False, "optimization_gain_verified": False,
        **failure(code, _MESSAGES[code]),
    }
