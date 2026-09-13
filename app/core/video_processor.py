import gc
import os
import shutil
import tempfile
from datetime import datetime
from typing import List, Dict, Optional

from app.config import settings
from app.utils.logger import log
from app.utils.ffmpeg_utils import FFmpegUtils
from app.core.audio_analyzer import AudioAnalyzer
from app.ml.models.yolo_model import CricketYOLOModel
from app.ml.models.classifier_model import (
    CricketEventClassifier
)
from app.services.s3_service import S3Service


class CricketVideoProcessor:
    """
    Main video processing pipeline for CricCircle.

    Pipeline:
    1. Download video from S3
    2. Validate video (duration, format)
    3. Extract audio -> analyze for excitement spikes
    4. Extract frames -> run YOLO object detection
    5. Combine audio + visual detections
    6. Classify each detected moment
    7. Cut highlight clips (6 seconds each)
    8. Add CricCircle branding
    9. Generate thumbnails
    10. Upload to S3
    11. Return structured results
    """

    def __init__(self):
        self.ffmpeg = FFmpegUtils()
        self.audio_analyzer = AudioAnalyzer()
        self.s3_service = S3Service()
        # YOLO and the ResNet50 classifier are loaded lazily (see the
        # properties below) instead of here. This class is instantiated
        # once at app startup as a module-level singleton, so eager
        # construction meant both models -- torch nn.Modules, the
        # classifier alone is ~100MB of float32 weights -- sat resident
        # in memory for the entire life of the process, whether or not
        # a video was ever processed. On a fixed 512MB host that ate
        # into the headroom the (also memory-hungry) audio-analysis
        # step needed, and contributed to an OOM kill mid-request.
        self._yolo_model: Optional[CricketYOLOModel] = None
        self._classifier: Optional[CricketEventClassifier] = None

        self.clip_duration = (
            settings.highlight_clip_duration
        )
        self.frames_per_second = settings.frames_per_second
        self.min_confidence = settings.confidence_threshold

    @property
    def yolo_model(self) -> CricketYOLOModel:
        if self._yolo_model is None:
            self._yolo_model = CricketYOLOModel()
        return self._yolo_model

    @property
    def classifier(self) -> CricketEventClassifier:
        if self._classifier is None:
            self._classifier = CricketEventClassifier()
        return self._classifier

    async def process(
        self,
        video_s3_key: str,
        video_id: str,
        user_id: str,
        match_id: Optional[str] = None,
        progress_callback=None
    ) -> Dict:
        """
        Main processing function.
        Returns list of detected highlights.
        """
        temp_dir = None
        start_time = datetime.utcnow()

        try:
            # Create temp workspace
            temp_dir = tempfile.mkdtemp(
                prefix=f"criccircle_{video_id}_"
            )
            log.info(
                f"Processing video {video_id} "
                f"for user {user_id}"
            )

            await self._update_progress(
                progress_callback, 5,
                "Downloading video..."
            )

            # STEP 1: Download video from S3
            video_path = os.path.join(
                temp_dir, "input_video.mp4"
            )
            await self.s3_service.download_file(
                video_s3_key, video_path
            )
            log.info(f"Video downloaded to {video_path}")

            await self._update_progress(
                progress_callback, 10,
                "Validating video..."
            )

            # STEP 2: Get video info and validate
            self.ffmpeg.get_video_info(
                video_path
            )
            duration = self.ffmpeg.get_video_duration(
                video_path
            )

            if duration > settings.max_video_duration_seconds:
                raise ValueError(
                    f"Video too long: {duration}s "
                    f"(max {settings.max_video_duration_seconds}s)"
                )

            log.info(
                f"Video duration: {duration:.1f}s"
            )

            await self._update_progress(
                progress_callback, 20,
                "Analyzing audio..."
            )

            # STEP 3: Extract and analyze audio
            audio_path = os.path.join(
                temp_dir, "audio.wav"
            )
            self.ffmpeg.extract_audio(
                video_path, audio_path
            )
            audio_detections = self.audio_analyzer.analyze(
                audio_path
            )
            log.info(
                f"Audio found {len(audio_detections)} "
                f"potential moments"
            )

            # analyze() builds several full-length waveform/spectrogram
            # arrays that go out of scope on return; force a prompt
            # collect before the YOLO model (loaded lazily, right
            # below) adds its own footprint on top -- this handoff is
            # what was OOM-killing the process on 512MB hosts.
            gc.collect()

            await self._update_progress(
                progress_callback, 40,
                "Detecting cricket moments..."
            )

            # STEP 4: Extract frames and run YOLO
            frames_dir = os.path.join(temp_dir, "frames")
            frames = self.ffmpeg.extract_frames(
                video_path,
                frames_dir,
                fps=self.frames_per_second
            )

            log.info(
                f"Extracted {len(frames)} frames"
            )

            # Run YOLO on all frames in batches
            frame_detections = self.yolo_model.detect_batch(
                frames, batch_size=16
            )
            gc.collect()

            await self._update_progress(
                progress_callback, 60,
                "Identifying highlight moments..."
            )

            # STEP 5: Find highlight timestamps
            visual_highlights = (
                self._find_visual_highlights(
                    frames,
                    frame_detections,
                    duration
                )
            )

            # STEP 6: Combine audio + visual
            combined = self._combine_detections(
                audio_detections,
                visual_highlights,
                duration
            )

            log.info(
                f"Combined: {len(combined)} highlights"
            )

            await self._update_progress(
                progress_callback, 70,
                "Classifying highlight types..."
            )

            # STEP 7: Classify each highlight
            classified = await self._classify_highlights(
                combined,
                video_path,
                frames_dir,
                temp_dir
            )

            await self._update_progress(
                progress_callback, 85,
                "Creating highlight clips..."
            )

            # STEP 8: Cut clips and add branding
            final_highlights = await self._create_clips(
                classified,
                video_path,
                temp_dir,
                user_id,
                video_id
            )

            await self._update_progress(
                progress_callback, 95,
                "Uploading highlights..."
            )

            # Calculate processing time
            processing_time = (
                datetime.utcnow() - start_time
            ).total_seconds()

            log.info(
                f"Processing complete for {video_id}. "
                f"Found {len(final_highlights)} highlights "
                f"in {processing_time:.1f}s"
            )

            await self._update_progress(
                progress_callback, 100,
                "Complete!"
            )

            return {
                "video_id": video_id,
                "user_id": user_id,
                "match_id": match_id,
                "duration_seconds": duration,
                "highlights": final_highlights,
                "total_found": len(final_highlights),
                "processing_duration_seconds": processing_time,
                "status": "completed"
            }

        except Exception as e:
            log.error(
                f"Processing failed for {video_id}: {str(e)}"
            )
            return {
                "video_id": video_id,
                "user_id": user_id,
                "status": "failed",
                "error": str(e),
                "highlights": []
            }
        finally:
            # Clean up temp files
            if temp_dir and os.path.exists(temp_dir):
                shutil.rmtree(temp_dir, ignore_errors=True)

    def _find_visual_highlights(
        self,
        frames: List[str],
        detections: List[Dict],
        video_duration: float
    ) -> List[Dict]:
        """
        Analyze YOLO detections to find highlight moments.

        Highlight indicators:
        - Ball detected near boundary line
        - Stumps disturbed (ball + stumps close together)
        - Multiple players in celebration pose
        """
        highlights = []

        for i, (frame_path, detection) in enumerate(
            zip(frames, detections)
        ):
            objects = detection.get("objects", [])
            timestamp = i / self.frames_per_second

            obj_names = [
                obj["class_name"] for obj in objects
            ]
            obj_dict = {}
            for obj in objects:
                name = obj["class_name"]
                if (name not in obj_dict or
                    obj["confidence"] >
                    obj_dict[name]["confidence"]):
                    obj_dict[name] = obj

            # Pattern 1: Ball near boundary
            if (
                "cricket_ball" in obj_dict and
                "boundary_rope" in obj_dict
            ):
                highlights.append({
                    "timestamp": max(0, timestamp - 2),
                    "peak_timestamp": timestamp,
                    "confidence": 0.75,
                    "method": "ball_boundary",
                    "hint_type": "boundary_shot",
                })

            # Pattern 2: Stumps disturbed (ball + stumps)
            if (
                "cricket_ball" in obj_dict and
                "stumps" in obj_dict
            ):
                ball_bbox = obj_dict[
                    "cricket_ball"
                ]["bbox"]
                stump_bbox = obj_dict["stumps"]["bbox"]

                ball_x = (
                    ball_bbox["x1"] + ball_bbox["x2"]
                ) / 2
                stump_x = (
                    stump_bbox["x1"] + stump_bbox["x2"]
                ) / 2
                distance = abs(ball_x - stump_x)

                if distance < 100:  # pixels
                    highlights.append({
                        "timestamp": max(
                            0, timestamp - 1
                        ),
                        "peak_timestamp": timestamp,
                        "confidence": 0.85,
                        "method": "stumps_disturbed",
                        "hint_type": "wicket",
                    })

            # Pattern 3: Multiple fielders gathered
            fielder_count = obj_names.count("fielder")
            if fielder_count >= 3:
                highlights.append({
                    "timestamp": max(0, timestamp - 2),
                    "peak_timestamp": timestamp,
                    "confidence": 0.65,
                    "method": "player_cluster",
                    "hint_type": "celebration",
                })

        return highlights

    def _combine_detections(
        self,
        audio_detections: List[Dict],
        visual_detections: List[Dict],
        duration: float
    ) -> List[Dict]:
        """
        Combine audio and visual detections.
        When both agree within 3 seconds = higher confidence.
        """
        combined = []
        used_audio = set()

        for vis in visual_detections:
            vis_ts = vis["timestamp"]

            matching_audio = None
            for i, aud in enumerate(audio_detections):
                if i in used_audio:
                    continue
                aud_ts = aud["timestamp"]
                if abs(vis_ts - aud_ts) <= 3.0:
                    matching_audio = (i, aud)
                    break

            if matching_audio:
                idx, aud = matching_audio
                used_audio.add(idx)

                combined_confidence = min(
                    1.0,
                    vis["confidence"] * 0.5 +
                    aud["confidence"] * 0.5 + 0.15
                )

                combined.append({
                    "timestamp": vis_ts,
                    "peak_timestamp": vis[
                        "peak_timestamp"
                    ],
                    "confidence": combined_confidence,
                    "method": "audio_visual_combined",
                    "hint_type": vis.get(
                        "hint_type", "unknown"
                    ),
                    "audio_confidence": aud["confidence"],
                    "visual_confidence": vis["confidence"],
                })
            else:
                combined.append(vis)

        # Add audio-only detections not matched to visual
        for i, aud in enumerate(audio_detections):
            if i not in used_audio:
                combined.append({
                    **aud,
                    "method": "audio_only",
                    "hint_type": "unknown",
                })

        # Filter by minimum confidence
        filtered = [
            d for d in combined
            if d["confidence"] >= self.min_confidence
        ]

        # Deduplicate within 4 seconds
        filtered = self._deduplicate(filtered, 4.0)

        filtered.sort(key=lambda x: x["timestamp"])

        log.info(
            f"Combined detections: {len(filtered)} "
            f"highlights (min confidence: "
            f"{self.min_confidence})"
        )

        return filtered

    async def _classify_highlights(
        self,
        highlights: List[Dict],
        video_path: str,
        frames_dir: str,
        temp_dir: str
    ) -> List[Dict]:
        """Classify each highlight using ResNet50"""
        classified = []

        for highlight in highlights:
            start_ts = highlight["timestamp"]

            clip_frames = self._get_frames_for_timestamp(
                frames_dir,
                start_ts,
                duration=self.clip_duration
            )

            classification = (
                self.classifier.classify_clip_frames(
                    clip_frames
                )
            )

            event_type = classification["event_type"]

            if (
                classification["confidence"] < 0.6 and
                highlight.get("hint_type") != "unknown"
            ):
                event_type = highlight.get(
                    "hint_type", event_type
                )

            if event_type == "normal_play":
                continue

            classified.append({
                **highlight,
                "event_type": event_type,
                "classifier_confidence": (
                    classification["confidence"]
                ),
                "display_name": (
                    CricketEventClassifier.DISPLAY_NAMES.get(
                        event_type, "Unknown"
                    )
                ),
                "all_scores": classification.get(
                    "all_scores", {}
                ),
            })

        return classified

    async def _create_clips(
        self,
        highlights: List[Dict],
        video_path: str,
        temp_dir: str,
        user_id: str,
        video_id: str
    ) -> List[Dict]:
        """Cut and process highlight clips"""
        final = []
        clips_dir = os.path.join(temp_dir, "clips")
        os.makedirs(clips_dir, exist_ok=True)

        for i, highlight in enumerate(highlights):
            try:
                ts = highlight["timestamp"]
                event_type = highlight["event_type"]

                # 1. Cut raw clip
                raw_clip = os.path.join(
                    clips_dir, f"raw_{i}_{event_type}.mp4"
                )
                self.ffmpeg.cut_clip(
                    video_path,
                    start_seconds=ts,
                    duration=self.clip_duration,
                    output_path=raw_clip
                )

                # 2. Generate thumbnail
                thumbnail_path = os.path.join(
                    clips_dir,
                    f"thumb_{i}_{event_type}.jpg"
                )
                self.ffmpeg.generate_thumbnail(
                    raw_clip,
                    timestamp_seconds=2.0,
                    output_path=thumbnail_path
                )

                # 3. Add branding
                branded_clip = os.path.join(
                    clips_dir,
                    f"branded_{i}_{event_type}.mp4"
                )

                if (
                    settings.branding_enabled and
                    os.path.exists(settings.watermark_path)
                ):
                    self.ffmpeg.add_watermark(
                        raw_clip,
                        settings.watermark_path,
                        branded_clip
                    )
                else:
                    shutil.copy(raw_clip, branded_clip)

                # 4. Upload to S3
                s3_clip_key = (
                    f"highlights/{user_id}/"
                    f"{video_id}/clip_{i}_{event_type}.mp4"
                )
                s3_thumb_key = (
                    f"highlights/{user_id}/"
                    f"{video_id}/thumb_{i}_{event_type}.jpg"
                )

                clip_url = await self.s3_service.upload_file(
                    branded_clip, s3_clip_key,
                    content_type="video/mp4"
                )
                thumb_url = await self.s3_service.upload_file(
                    thumbnail_path, s3_thumb_key,
                    content_type="image/jpeg"
                )

                # 5. Build final highlight object
                final.append({
                    "id": f"{video_id}_highlight_{i}",
                    "event_type": event_type,
                    "display_name": highlight.get(
                        "display_name", "Highlight"
                    ),
                    "timestamp_seconds": ts,
                    "peak_timestamp": highlight.get(
                        "peak_timestamp", ts
                    ),
                    "duration_seconds": self.clip_duration,
                    "confidence_score": highlight.get(
                        "confidence", 0.0
                    ),
                    "classifier_confidence": highlight.get(
                        "classifier_confidence", 0.0
                    ),
                    "clip_url": clip_url,
                    "thumbnail_url": thumb_url,
                    "clip_s3_key": s3_clip_key,
                    "detection_method": highlight.get(
                        "method", "unknown"
                    ),
                    "status": "ai_detected",
                    "all_scores": highlight.get(
                        "all_scores", {}
                    ),
                })

                log.info(
                    f"Clip {i}: {event_type} at "
                    f"{ts:.1f}s "
                    f"(confidence: "
                    f"{highlight.get('confidence', 0):.2f})"
                )

            except Exception as e:
                log.error(
                    f"Failed to process clip {i}: {e}"
                )
                continue

        return final

    def _get_frames_for_timestamp(
        self,
        frames_dir: str,
        timestamp: float,
        duration: float
    ) -> List[str]:
        """Get frame files for a specific time range"""
        start_frame = int(
            timestamp * self.frames_per_second
        )
        end_frame = int(
            (timestamp + duration) * self.frames_per_second
        )

        all_frames = sorted([
            os.path.join(frames_dir, f)
            for f in os.listdir(frames_dir)
            if f.endswith(".jpg")
        ])

        selected = all_frames[start_frame:end_frame]
        return selected

    def _deduplicate(
        self,
        detections: List[Dict],
        min_gap: float
    ) -> List[Dict]:
        """Remove detections within min_gap seconds"""
        if not detections:
            return []

        detections.sort(
            key=lambda x: x["confidence"],
            reverse=True
        )

        filtered = []
        used_timestamps = []

        for det in detections:
            ts = det["timestamp"]
            too_close = any(
                abs(ts - used_ts) < min_gap
                for used_ts in used_timestamps
            )
            if not too_close:
                filtered.append(det)
                used_timestamps.append(ts)

        filtered.sort(key=lambda x: x["timestamp"])
        return filtered

    async def _update_progress(
        self,
        callback,
        percentage: int,
        message: str
    ):
        """Update processing progress"""
        if callback:
            await callback(percentage, message)
        log.info(f"Progress {percentage}%: {message}")
