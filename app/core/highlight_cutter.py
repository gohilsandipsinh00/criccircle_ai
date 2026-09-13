import os
import shutil
from app.config import settings
from app.utils.ffmpeg_utils import FFmpegUtils
from app.utils.logger import log


class HighlightCutter:
    """
    Cuts a single highlight clip from the source video, generates
    a thumbnail, and applies CricCircle branding (watermark).
    """

    def __init__(self):
        self.clip_duration = settings.highlight_clip_duration

    def cut_and_brand(
        self,
        video_path: str,
        start_seconds: float,
        output_dir: str,
        clip_name: str
    ) -> dict:
        os.makedirs(output_dir, exist_ok=True)

        raw_clip = os.path.join(output_dir, f"raw_{clip_name}.mp4")
        thumbnail = os.path.join(output_dir, f"thumb_{clip_name}.jpg")
        branded_clip = os.path.join(output_dir, f"branded_{clip_name}.mp4")

        FFmpegUtils.cut_clip(
            video_path,
            start_seconds=start_seconds,
            duration=self.clip_duration,
            output_path=raw_clip
        )

        FFmpegUtils.generate_thumbnail(
            raw_clip, timestamp_seconds=2.0, output_path=thumbnail
        )

        if settings.branding_enabled and os.path.exists(
            settings.watermark_path
        ):
            FFmpegUtils.add_watermark(
                raw_clip, settings.watermark_path, branded_clip
            )
        else:
            shutil.copy(raw_clip, branded_clip)

        log.info(f"Clip cut and branded: {branded_clip}")

        return {
            "raw_clip": raw_clip,
            "branded_clip": branded_clip,
            "thumbnail": thumbnail,
        }
