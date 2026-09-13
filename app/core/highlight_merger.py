from typing import List, Dict


class HighlightMerger:
    """
    Combines audio-based and visual (YOLO) detections into a single
    ranked list of candidate highlight moments, boosting confidence
    when both signals agree on roughly the same timestamp.
    """

    def __init__(self, agreement_window: float = 3.0, min_gap: float = 4.0):
        self.agreement_window = agreement_window
        self.min_gap = min_gap

    def merge(
        self,
        audio_detections: List[Dict],
        visual_detections: List[Dict],
        min_confidence: float = 0.0
    ) -> List[Dict]:
        combined = []
        used_audio = set()

        for vis in visual_detections:
            vis_ts = vis["timestamp"]

            matching_audio = None
            for i, aud in enumerate(audio_detections):
                if i in used_audio:
                    continue
                if abs(vis_ts - aud["timestamp"]) <= self.agreement_window:
                    matching_audio = (i, aud)
                    break

            if matching_audio:
                idx, aud = matching_audio
                used_audio.add(idx)

                combined_confidence = min(
                    1.0,
                    vis["confidence"] * 0.5 + aud["confidence"] * 0.5 + 0.15
                )

                combined.append({
                    "timestamp": vis_ts,
                    "peak_timestamp": vis["peak_timestamp"],
                    "confidence": combined_confidence,
                    "method": "audio_visual_combined",
                    "hint_type": vis.get("hint_type", "unknown"),
                    "audio_confidence": aud["confidence"],
                    "visual_confidence": vis["confidence"],
                })
            else:
                combined.append(vis)

        for i, aud in enumerate(audio_detections):
            if i not in used_audio:
                combined.append({
                    **aud,
                    "method": "audio_only",
                    "hint_type": "unknown",
                })

        filtered = [
            d for d in combined if d["confidence"] >= min_confidence
        ]
        filtered = self._deduplicate(filtered, self.min_gap)
        filtered.sort(key=lambda x: x["timestamp"])
        return filtered

    def _deduplicate(
        self, detections: List[Dict], min_gap: float
    ) -> List[Dict]:
        if not detections:
            return []

        detections.sort(key=lambda x: x["confidence"], reverse=True)

        filtered = []
        used_timestamps = []
        for det in detections:
            ts = det["timestamp"]
            too_close = any(
                abs(ts - used_ts) < min_gap for used_ts in used_timestamps
            )
            if not too_close:
                filtered.append(det)
                used_timestamps.append(ts)

        filtered.sort(key=lambda x: x["timestamp"])
        return filtered
