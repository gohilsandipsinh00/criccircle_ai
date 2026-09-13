import subprocess
import os
import json
from typing import List
from app.utils.logger import log


class FFmpegUtils:

    @staticmethod
    def get_video_info(video_path: str) -> dict:
        """Get video metadata using ffprobe"""
        cmd = [
            "ffprobe",
            "-v", "quiet",
            "-print_format", "json",
            "-show_format",
            "-show_streams",
            video_path
        ]
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True
        )
        if result.returncode != 0:
            raise Exception(
                f"FFprobe error: {result.stderr}"
            )
        return json.loads(result.stdout)

    @staticmethod
    def extract_frames(
        video_path: str,
        output_dir: str,
        fps: float = 2.0,
        quality: int = 2
    ) -> List[str]:
        """Extract frames from video at specified FPS"""
        os.makedirs(output_dir, exist_ok=True)
        output_pattern = os.path.join(
            output_dir,
            "frame_%06d.jpg"
        )
        cmd = [
            "ffmpeg",
            "-i", video_path,
            "-vf", f"fps={fps}",
            "-q:v", str(quality),
            "-y",
            output_pattern
        ]
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True
        )
        if result.returncode != 0:
            raise Exception(
                f"Frame extraction error: {result.stderr}"
            )
        frames = sorted([
            os.path.join(output_dir, f)
            for f in os.listdir(output_dir)
            if f.endswith(".jpg")
        ])
        log.info(
            f"Extracted {len(frames)} frames from {video_path}"
        )
        return frames

    @staticmethod
    def extract_audio(
        video_path: str,
        output_path: str
    ) -> str:
        """Extract audio track from video"""
        cmd = [
            "ffmpeg",
            "-i", video_path,
            "-vn",
            "-acodec", "pcm_s16le",
            "-ar", "44100",
            "-ac", "2",
            "-y",
            output_path
        ]
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True
        )
        if result.returncode != 0:
            raise Exception(
                f"Audio extraction error: {result.stderr}"
            )
        log.info(f"Audio extracted to {output_path}")
        return output_path

    @staticmethod
    def cut_clip(
        video_path: str,
        start_seconds: float,
        duration: float,
        output_path: str
    ) -> str:
        """Cut a clip from video"""
        cmd = [
            "ffmpeg",
            "-ss", str(start_seconds),
            "-i", video_path,
            "-t", str(duration),
            "-c:v", "libx264",
            "-c:a", "aac",
            "-preset", "fast",
            "-crf", "23",
            "-y",
            output_path
        ]
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True
        )
        if result.returncode != 0:
            raise Exception(
                f"Clip cutting error: {result.stderr}"
            )
        return output_path

    @staticmethod
    def add_watermark(
        video_path: str,
        watermark_path: str,
        output_path: str,
        position: str = "bottom_right"
    ) -> str:
        """Add CricCircle watermark to video"""
        positions = {
            "top_left": "10:10",
            "top_right": "main_w-overlay_w-10:10",
            "bottom_left": "10:main_h-overlay_h-10",
            "bottom_right": (
                "main_w-overlay_w-10:main_h-overlay_h-10"
            ),
        }
        pos = positions.get(position, "10:10")

        cmd = [
            "ffmpeg",
            "-i", video_path,
            "-i", watermark_path,
            "-filter_complex",
            (
                f"[1:v]scale=120:-1[wm];"
                f"[0:v][wm]overlay={pos}"
            ),
            "-codec:a", "copy",
            "-y",
            output_path
        ]
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True
        )
        if result.returncode != 0:
            raise Exception(
                f"Watermark error: {result.stderr}"
            )
        return output_path

    @staticmethod
    def add_text_overlay(
        video_path: str,
        text: str,
        output_path: str,
        font_size: int = 24,
        position: str = "bottom"
    ) -> str:
        """Add text overlay (player name, category)"""
        if position == "bottom":
            y_pos = "h-th-20"
        else:
            y_pos = "20"

        safe_text = text.replace("'", "").replace(":", "")

        cmd = [
            "ffmpeg",
            "-i", video_path,
            "-vf",
            (
                f"drawtext=text='{safe_text}':"
                f"fontsize={font_size}:"
                f"fontcolor=white:"
                f"x=(w-tw)/2:"
                f"y={y_pos}:"
                f"box=1:"
                f"boxcolor=black@0.5:"
                f"boxborderw=5"
            ),
            "-codec:a", "copy",
            "-y",
            output_path
        ]
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True
        )
        if result.returncode != 0:
            raise Exception(
                f"Text overlay error: {result.stderr}"
            )
        return output_path

    @staticmethod
    def generate_thumbnail(
        video_path: str,
        timestamp_seconds: float,
        output_path: str,
        width: int = 640,
        height: int = 360
    ) -> str:
        """Generate thumbnail at specific timestamp"""
        cmd = [
            "ffmpeg",
            "-ss", str(timestamp_seconds),
            "-i", video_path,
            "-vframes", "1",
            "-vf", f"scale={width}:{height}",
            "-y",
            output_path
        ]
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True
        )
        if result.returncode != 0:
            raise Exception(
                f"Thumbnail error: {result.stderr}"
            )
        return output_path

    @staticmethod
    def get_video_duration(video_path: str) -> float:
        """Get video duration in seconds"""
        cmd = [
            "ffprobe",
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            video_path
        ]
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True
        )
        return float(result.stdout.strip())
