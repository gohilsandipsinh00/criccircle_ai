"""
Turns human feedback (confirm/reject/edit) collected via the
cric_video_ai_train app into a classifier training dataset in the
folder layout app/ml/training/prepare_dataset.py expects:

  data/annotated/classifier/<class_name>/*.jpg

- confirm  -> frames from the highlight's own [start, start+duration)
              range, labeled with its AI-assigned event_type.
- edit     -> frames from the corrected [start, end) range if the
              user trimmed it, else the original range, labeled with
              corrected_type if the user reclassified it, else the
              original event_type.
- reject   -> frames from the original detection range, labeled
              "false_positive" -- a negative example: something that
              looked like a highlight to the model but wasn't one.

Requires ffmpeg/ffprobe on PATH (same requirement as the main
pipeline). Re-downloads/copies each source video once per run via
S3Service, which transparently handles both local-disk and real S3
storage depending on settings.use_s3.

Usage:
  python scripts/export_feedback_to_dataset.py \
    --output data/annotated/classifier --fps 2
"""
import argparse
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services.local_store import store  # noqa: E402
from app.services.s3_service import S3Service  # noqa: E402
from app.utils.ffmpeg_utils import FFmpegUtils  # noqa: E402


def latest_feedback_per_highlight(events: List[Dict]) -> Dict[str, Dict]:
    """Keep only each highlight's most recent feedback event."""
    latest: Dict[str, Dict] = {}
    for e in events:
        latest[e["highlight_id"]] = e
    return latest


def resolve_label_and_range(event: Dict, highlight: Dict) -> Optional[tuple]:
    """Returns (class_name, start_seconds, end_seconds) or None to skip."""
    action = event["action"]
    original_start = highlight["timestamp_seconds"]
    original_end = original_start + highlight.get("duration_seconds", 6)

    if action == "reject":
        return "false_positive", original_start, original_end

    if action == "confirm":
        return highlight["event_type"], original_start, original_end

    if action == "edit":
        class_name = event.get("corrected_type") or highlight["event_type"]
        start = event.get("corrected_start_seconds")
        end = event.get("corrected_end_seconds")
        if start is None:
            start = original_start
        if end is None:
            end = original_end
        if end <= start:
            return None
        return class_name, start, end

    return None


def export(output_dir: str, fps: float):
    events = store.read_feedback()
    if not events:
        print("No feedback recorded yet -- nothing to export.")
        return

    reviewed = latest_feedback_per_highlight(events)
    print(f"{len(reviewed)} reviewed highlights across {len(events)} events")

    s3 = S3Service()
    output = Path(output_dir)
    video_cache: Dict[str, str] = {}
    written = 0
    skipped = 0

    with tempfile.TemporaryDirectory(prefix="criccircle_export_") as tmp:
        for highlight_id, event in reviewed.items():
            video_id = event["video_id"]
            highlight = store.find_highlight(highlight_id)
            if not highlight:
                print(f"  skip {highlight_id}: highlight record missing")
                skipped += 1
                continue

            resolved = resolve_label_and_range(event, highlight)
            if not resolved:
                print(f"  skip {highlight_id}: no valid label/time range")
                skipped += 1
                continue
            class_name, start, end = resolved

            status = store.get_status(video_id)
            raw_key = status.get("raw_video_s3_key") if status else None
            if not raw_key:
                print(f"  skip {highlight_id}: source video unknown "
                      f"for {video_id}")
                skipped += 1
                continue

            if video_id not in video_cache:
                local_path = str(Path(tmp) / f"{video_id}.mp4")
                try:
                    s3.download_file(raw_key, local_path)
                except Exception as e:
                    print(f"  skip video {video_id}: fetch failed ({e})")
                    video_cache[video_id] = ""
                video_cache[video_id] = local_path

            video_path = video_cache.get(video_id)
            if not video_path:
                skipped += 1
                continue

            class_dir = output / class_name
            class_dir.mkdir(parents=True, exist_ok=True)

            clip_path = Path(tmp) / f"{highlight_id}_clip.mp4"
            frames_dir = Path(tmp) / f"{highlight_id}_frames"
            try:
                FFmpegUtils.cut_clip(
                    video_path,
                    start_seconds=max(0, start),
                    duration=max(0.5, end - start),
                    output_path=str(clip_path),
                )
                frames = FFmpegUtils.extract_frames(
                    str(clip_path), str(frames_dir), fps=fps
                )
            except Exception as e:
                print(f"  skip {highlight_id}: ffmpeg failed ({e})")
                skipped += 1
                continue

            for i, frame in enumerate(frames):
                dest = class_dir / f"{video_id}_{highlight_id}_{i:03d}.jpg"
                shutil.copy(frame, dest)
                written += 1

            print(
                f"  {highlight_id}: {event['action']} -> {class_name} "
                f"({len(frames)} frames from {start:.1f}s-{end:.1f}s)"
            )

    print(f"\nDone. {written} frames written, {skipped} highlights skipped.")
    print(f"Dataset at: {output}")
    print(
        "Next: python app/ml/training/prepare_dataset.py "
        f"--source {output} --output dataset/classifier"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/annotated/classifier")
    parser.add_argument("--fps", type=float, default=2.0)
    args = parser.parse_args()

    export(args.output, args.fps)
