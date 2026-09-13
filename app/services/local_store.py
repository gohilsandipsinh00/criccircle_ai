"""
File-backed persistence used while Postgres isn't provisioned yet
(Alembic migrations exist under app/models but haven't been applied
against a real DATABASE_URL -- see app/database.py).

Everything here is scoped to a single process and guarded by an
in-process lock, which is fine for a demo/labeling tool but not for
concurrent multi-instance deployment. When a real Postgres is
available, swap the method bodies below for SQLAlchemy reads/writes
against the existing Video / Highlight / ProcessingJob models -- the
call sites in app/api/routes/*.py only depend on this class's public
method signatures, so the swap doesn't ripple outward.

On-disk layout (under settings.local_data_dir):
  raw/                              uploaded source videos (local upload fallback)
  highlights/                       local "S3" output for clips + thumbnails
  processed/status/<video_id>.json  latest processing status per video
  processed/highlights/<video_id>.json  detected highlights per video
  processed/highlight_index.json    highlight_id -> video_id lookup
  processed/feedback_log.jsonl      append-only human feedback events
"""
import json
import threading
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from app.config import settings

_lock = threading.Lock()


class LocalStore:
    def __init__(self, base_dir: Optional[str] = None):
        self.base = Path(base_dir or settings.local_data_dir)
        self.raw_dir = self.base / "raw"
        self.highlights_output_dir = self.base / "highlights"
        self.processed_dir = self.base / "processed"
        self.status_dir = self.processed_dir / "status"
        self.highlights_dir = self.processed_dir / "highlights"
        self.highlight_index_path = (
            self.processed_dir / "highlight_index.json"
        )
        self.feedback_log_path = (
            self.processed_dir / "feedback_log.jsonl"
        )

        for d in (
            self.raw_dir,
            self.highlights_output_dir,
            self.status_dir,
            self.highlights_dir,
        ):
            d.mkdir(parents=True, exist_ok=True)

    # ---------------- video status ----------------

    def save_status(
        self,
        video_id: str,
        status: str,
        progress_percentage: float,
        current_step: str,
        user_id: Optional[str] = None,
        raw_video_s3_key: Optional[str] = None,
        error: Optional[str] = None,
    ):
        path = self.status_dir / f"{video_id}.json"
        existing = {}
        if path.exists():
            existing = json.loads(path.read_text())

        data = {
            **existing,
            "video_id": video_id,
            "status": status,
            "progress_percentage": progress_percentage,
            "current_step": current_step,
            "updated_at": datetime.utcnow().isoformat(),
        }
        if user_id is not None:
            data["user_id"] = user_id
        if raw_video_s3_key is not None:
            data["raw_video_s3_key"] = raw_video_s3_key
        if error is not None:
            data["error"] = error

        with _lock:
            path.write_text(json.dumps(data, indent=2))

    def get_status(self, video_id: str) -> Optional[Dict]:
        path = self.status_dir / f"{video_id}.json"
        if not path.exists():
            return None
        return json.loads(path.read_text())

    # ---------------- highlights ----------------

    def save_highlights(self, video_id: str, highlights: List[Dict]):
        path = self.highlights_dir / f"{video_id}.json"
        stored = []
        for h in highlights:
            stored.append({
                **h,
                "review_status": "ai_detected",
                "corrected_type": None,
                "corrected_start_seconds": None,
                "corrected_end_seconds": None,
                "quality_rating": None,
                "notes": None,
            })

        with _lock:
            path.write_text(json.dumps(stored, indent=2))
            self._index_highlights(video_id, stored)

    def _index_highlights(self, video_id: str, highlights: List[Dict]):
        index = {}
        if self.highlight_index_path.exists():
            index = json.loads(self.highlight_index_path.read_text())
        for h in highlights:
            index[h["id"]] = video_id
        self.highlight_index_path.write_text(json.dumps(index, indent=2))

    def get_highlights(self, video_id: str) -> List[Dict]:
        path = self.highlights_dir / f"{video_id}.json"
        if not path.exists():
            return []
        return json.loads(path.read_text())

    def find_highlight(self, highlight_id: str) -> Optional[Dict]:
        """Returns (video_id, highlight_dict) or None."""
        if not self.highlight_index_path.exists():
            return None
        index = json.loads(self.highlight_index_path.read_text())
        video_id = index.get(highlight_id)
        if not video_id:
            return None
        for h in self.get_highlights(video_id):
            if h["id"] == highlight_id:
                return {"video_id": video_id, **h}
        return None

    def update_highlight(self, highlight_id: str, **fields) -> Optional[Dict]:
        found = self.find_highlight(highlight_id)
        if not found:
            return None
        video_id = found["video_id"]
        path = self.highlights_dir / f"{video_id}.json"

        with _lock:
            highlights = json.loads(path.read_text())
            updated = None
            for h in highlights:
                if h["id"] == highlight_id:
                    h.update(fields)
                    updated = h
                    break
            path.write_text(json.dumps(highlights, indent=2))

        return updated

    # ---------------- feedback ----------------

    def append_feedback(self, event: Dict):
        event = {"timestamp": datetime.utcnow().isoformat(), **event}
        with _lock:
            with open(self.feedback_log_path, "a") as f:
                f.write(json.dumps(event) + "\n")

    def read_feedback(self) -> List[Dict]:
        if not self.feedback_log_path.exists():
            return []
        events = []
        with open(self.feedback_log_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    events.append(json.loads(line))
        return events

    def get_stats(self) -> Dict:
        events = self.read_feedback()
        total = len(events)
        confirmed = sum(1 for e in events if e["action"] == "confirm")
        rejected = sum(1 for e in events if e["action"] == "reject")
        edited = sum(1 for e in events if e["action"] == "edit")

        by_type: Dict[str, Dict[str, int]] = {}
        for e in events:
            event_type = e.get("reviewed_type") or "unknown"
            bucket = by_type.setdefault(
                event_type, {"confirmed": 0, "rejected": 0, "edited": 0}
            )
            if e["action"] == "confirm":
                bucket["confirmed"] += 1
            elif e["action"] == "reject":
                bucket["rejected"] += 1
            elif e["action"] == "edit":
                bucket["edited"] += 1

        accuracy = (confirmed / total * 100) if total else 0.0

        return {
            "total_reviewed": total,
            "confirmed": confirmed,
            "rejected": rejected,
            "edited": edited,
            "accuracy_percent": round(accuracy, 1),
            "by_type": by_type,
        }


store = LocalStore()
