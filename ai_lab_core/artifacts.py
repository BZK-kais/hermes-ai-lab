"""Inspect and fingerprint outputs, independently of job exit codes."""
import hashlib
import os
from .integration_config import permitted_path
from .endpoint import failure


def inspect_artifact(path, settings, hash_files=False):
    try:
        root = permitted_path(path, settings)
        paths = [root] if root.is_file() else []
        if root.is_dir():
            visited = 0
            for directory, dirs, files in os.walk(root, followlinks=False):
                visited += len(dirs) + len(files)
                if visited > 10000:
                    return failure("artifact_limit", "Artifact contains too many entries to inspect.")
                dirs[:] = sorted(d for d in dirs if not (type(root)(directory) / d).is_symlink())
                paths.extend(type(root)(directory) / name for name in sorted(files))
        entries, total = [], 0
        if len(paths) > 10000:
            return failure("artifact_limit", "Artifact contains too many entries to inspect.")
        for file in paths:
            if not file.is_file():
                continue
            permitted_path(str(file), settings)
            size = file.stat().st_size
            total += size
            item = {"name": str(file.relative_to(root)) if root.is_dir() else file.name, "size_bytes": size}
            if hash_files:
                if total > 8 * 1024**3:
                    return failure("artifact_limit", "Hashing is limited to 8 GiB per request.")
                h = hashlib.sha256()
                with file.open("rb") as stream:
                    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                        h.update(chunk)
                item["sha256"] = h.hexdigest()
            entries.append(item)
        return {"ok": True, "path": str(root), "files": entries[:500], "file_count": len(entries), "total_size_bytes": total,
                "truncated": len(entries) > 500, "weights_load_verified": False}
    except (OSError, ValueError, TypeError):
        return failure("invalid_artifact", "Artifact must exist inside configured roots.")
