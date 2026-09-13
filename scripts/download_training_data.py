"""
Download cricket match videos from YouTube for building a
training dataset (requires yt-dlp: pip install yt-dlp).

Usage:
  python scripts/download_training_data.py \
    --urls urls.txt --output data/raw
"""
import argparse
import subprocess
from pathlib import Path


def download_videos(urls_file: str, output_dir: str):
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    with open(urls_file) as f:
        urls = [line.strip() for line in f if line.strip()]

    for i, url in enumerate(urls):
        print(f"[{i + 1}/{len(urls)}] Downloading {url}")
        subprocess.run([
            "yt-dlp",
            "-f", "best[height<=720]",
            "-o", f"{output_dir}/video_%(id)s.%(ext)s",
            url
        ])

    print(f"\nDownloaded {len(urls)} videos to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--urls", required=True)
    parser.add_argument("--output", default="data/raw")
    args = parser.parse_args()
    download_videos(args.urls, args.output)
