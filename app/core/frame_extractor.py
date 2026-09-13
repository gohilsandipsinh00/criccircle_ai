from typing import List
from app.config import settings
from app.utils.ffmpeg_utils import FFmpegUtils
from app.utils.logger import log


class FrameExtractor:
    """Thin facade over FFmpegUtils for the video processing pipeline."""

    def __init__(self, fps: float = None):
        self.fps = fps or settings.frames_per_second

    def extract(self, video_path: str, output_dir: str) -> List[str]:
        log.info(
            f"Extracting frames at {self.fps} fps from {video_path}"
        )
        return FFmpegUtils.extract_frames(
            video_path, output_dir, fps=self.fps
        )
