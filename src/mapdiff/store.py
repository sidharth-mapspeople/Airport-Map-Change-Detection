"""File-based version store.

data/<map_id>/
  versions/v0001/{source.<ext>, snapshot.json}
  changes/v0001_to_v0002.{json,md}
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path

from mapdiff.models import ChangeSet, Snapshot

_SAFE_ID = re.compile(r"^[A-Za-z0-9._-]+$")


class MapStore:
    def __init__(self, root: str | Path = "data"):
        self.root = Path(root)

    def _map_dir(self, map_id: str) -> Path:
        if not _SAFE_ID.match(map_id):
            raise ValueError(f"map_id may only contain letters, digits, '.', '_' and '-': {map_id!r}")
        return self.root / map_id

    def versions(self, map_id: str) -> list[int]:
        d = self._map_dir(map_id) / "versions"
        return sorted(int(p.name[1:]) for p in d.glob("v[0-9]*") if (p / "snapshot.json").exists()) if d.exists() else []

    def latest(self, map_id: str) -> Snapshot | None:
        vs = self.versions(map_id)
        return self.load(map_id, vs[-1]) if vs else None

    def _version_dir(self, map_id: str, version: int) -> Path:
        return self._map_dir(map_id) / "versions" / f"v{version:04d}"

    def load(self, map_id: str, version: int) -> Snapshot:
        return Snapshot.model_validate_json((self._version_dir(map_id, version) / "snapshot.json").read_text())

    def source_path(self, map_id: str, version: int) -> Path:
        return next(self._version_dir(map_id, version).glob("source.*"))

    def save(self, snapshot: Snapshot, source: Path) -> Path:
        """Assign the next version number and persist the snapshot with a copy of its source file."""
        vs = self.versions(snapshot.map_id)
        snapshot.version = (vs[-1] + 1) if vs else 1
        d = self._version_dir(snapshot.map_id, snapshot.version)
        d.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, d / f"source{source.suffix.lower()}")
        (d / "snapshot.json").write_text(snapshot.model_dump_json(indent=2))
        return d

    def save_changes(self, cs: ChangeSet, markdown: str) -> Path:
        d = self._map_dir(cs.map_id) / "changes"
        d.mkdir(parents=True, exist_ok=True)
        stem = d / f"v{cs.from_version:04d}_to_v{cs.to_version:04d}"
        stem.with_suffix(".json").write_text(cs.model_dump_json(indent=2))
        stem.with_suffix(".md").write_text(markdown)
        return stem.with_suffix(".md")
