"""
Persistent, crash-safe checkpoints for long-running collection jobs.

WHY THIS EXISTS
---------------
The existing resume_wikipedia_collection.py prints this about itself:

    "The current collector does not persist the API continuation token.
     Therefore we cannot safely resume from exactly page 759.
     We will restart the API traversal but preserve the existing data."

That works, but every resume replays the whole traversal from the beginning and
discards the pages it has already seen. On a job that has collected 100k of
170k documents, a restart costs 100k wasted requests before a single new
document arrives. On a source with rate limiting, that can be hours.

The fix is to persist the pagination cursor itself, not just the set of things
already fetched. This module does that, plus:

  - atomic writes (temp file + os.replace) so a kill during a checkpoint write
    cannot leave a half-written, unparseable checkpoint;
  - a seen-id set persisted alongside the cursor, so duplicates are impossible
    even if the cursor is stale or the source reorders results;
  - a monotonic `updated_at` timestamp, which is what the health monitor reads
    to distinguish "running and progressing" from "running but stalled".
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Optional


@dataclass
class CheckpointState:
    """Everything needed to resume a collection job exactly where it stopped."""

    job_name: str
    cursor: Optional[str] = None          # opaque, source-specific pagination token
    page: int = 0                         # how many pages/batches consumed
    collected: int = 0                    # documents successfully written
    skipped: int = 0                      # seen but rejected (filters, duplicates)
    errors: int = 0
    rate_limit_hits: int = 0
    started_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    finished: bool = False
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


class Checkpoint:
    """A resumable job checkpoint backed by two files.

    <name>.json      the CheckpointState (small, rewritten atomically)
    <name>.seen      newline-delimited ids already processed (append-only)

    The seen-file is append-only and flushed per write so that it is always at
    least as complete as the state file. Being *ahead* of the state file is safe
    (worst case we re-skip something); being behind would risk duplicates.
    """

    def __init__(self, path: str | Path, job_name: str = ""):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.seen_path = self.path.with_suffix(".seen")

        self.state = self._load_state(job_name or self.path.stem)
        self.seen: set[str] = self._load_seen()
        self._seen_fh = open(self.seen_path, "a", encoding="utf-8")

    def _load_state(self, job_name: str) -> CheckpointState:
        if self.path.exists():
            try:
                with open(self.path, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
                known = {f for f in CheckpointState.__dataclass_fields__}
                return CheckpointState(**{k: v for k, v in data.items() if k in known})
            except (json.JSONDecodeError, TypeError, ValueError):
                # Corrupt checkpoint: keep it for inspection, start clean.
                self.path.rename(self.path.with_suffix(".json.corrupt"))
        return CheckpointState(job_name=job_name)

    def _load_seen(self) -> set[str]:
        if not self.seen_path.exists():
            return set()
        with open(self.seen_path, "r", encoding="utf-8") as fh:
            return {line.strip() for line in fh if line.strip()}

    def has_seen(self, item_id: str) -> bool:
        return item_id in self.seen

    def mark_seen(self, item_id: str) -> None:
        """Record an id as processed. Durable before the caller does anything else."""
        if item_id in self.seen:
            return
        self.seen.add(item_id)
        self._seen_fh.write(item_id + "\n")
        self._seen_fh.flush()

    def save(self, **updates: Any) -> None:
        """Update counters/cursor and persist atomically."""
        for key, value in updates.items():
            if hasattr(self.state, key):
                setattr(self.state, key, value)
            else:
                self.state.extra[key] = value
        self.state.updated_at = time.time()

        fd, tmp = tempfile.mkstemp(dir=str(self.path.parent), suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(self.state.to_dict(), fh, ensure_ascii=False, indent=2)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, self.path)
        except BaseException:
            if os.path.exists(tmp):
                os.unlink(tmp)
            raise

    @property
    def age_seconds(self) -> float:
        """Seconds since the last checkpoint write. The staleness signal."""
        return time.time() - self.state.updated_at

    def close(self) -> None:
        self.save()
        if not self._seen_fh.closed:
            self._seen_fh.close()

    def __enter__(self) -> "Checkpoint":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


if __name__ == "__main__":
    import tempfile as _tf

    with _tf.TemporaryDirectory() as d:
        cp_path = Path(d) / "job.json"

        with Checkpoint(cp_path, "test_job") as cp:
            cp.mark_seen("a")
            cp.mark_seen("b")
            cp.save(cursor="CURSOR_1", page=1, collected=2)

        # Simulate a restart: a fresh object must recover cursor and seen-set.
        cp2 = Checkpoint(cp_path, "test_job")
        assert cp2.state.cursor == "CURSOR_1", cp2.state.cursor
        assert cp2.state.collected == 2
        assert cp2.has_seen("a") and cp2.has_seen("b")
        assert not cp2.has_seen("c")
        cp2.close()

        # A corrupt state file must not crash the job.
        cp_path.write_text("{ this is not json", encoding="utf-8")
        cp3 = Checkpoint(cp_path, "test_job")
        assert cp3.state.cursor is None
        assert cp3.has_seen("a"), "seen-set must survive a corrupt state file"
        assert cp_path.with_suffix(".json.corrupt").exists()
        cp3.close()

    print("checkpoint self-test: all assertions passed")
