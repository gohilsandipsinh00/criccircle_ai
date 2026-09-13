import shutil
from pathlib import Path

from app.config import settings
from app.utils.logger import log


class S3Service:
    """
    Storage abstraction with two backends, chosen by settings.use_s3:

    - S3 mode (use_s3=True): real boto3 calls against AWS, as in
      production.
    - Local mode (use_s3=False, the demo default -- no AWS creds
      available yet): "s3_key" is treated as a path relative to
      settings.local_data_dir, and download/upload become plain file
      copies. URLs are served back via the /media static mount
      registered in main.py.

    Callers (video_processor.py, the upload routes) don't need to
    know which mode is active.
    """

    def __init__(self):
        self.use_s3 = settings.use_s3
        self.highlights_bucket = settings.s3_bucket_highlights
        self.raw_bucket = settings.s3_bucket_raw_videos
        self.local_root = Path(settings.local_data_dir)

        if self.use_s3:
            import boto3
            self.client = boto3.client(
                "s3",
                aws_access_key_id=settings.aws_access_key_id,
                aws_secret_access_key=(
                    settings.aws_secret_access_key
                ),
                region_name=settings.aws_region,
            )
        else:
            self.client = None

    async def download_file(
        self,
        s3_key: str,
        local_path: str,
        bucket: str = None
    ) -> str:
        """Download/copy a file into local_path."""
        if not self.use_s3:
            src = self.local_root / s3_key
            log.info(f"[local] Copying {src} -> {local_path}")
            shutil.copy(src, local_path)
            return local_path

        bucket = bucket or self.raw_bucket
        log.info(
            f"Downloading s3://{bucket}/{s3_key}"
        )
        self.client.download_file(
            bucket, s3_key, local_path
        )
        return local_path

    async def upload_file(
        self,
        local_path: str,
        s3_key: str,
        bucket: str = None,
        content_type: str = "video/mp4"
    ) -> str:
        """Upload/copy a file, return a URL it can be fetched from."""
        if not self.use_s3:
            dest = self.local_root / s3_key
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(local_path, dest)
            url = f"{settings.public_base_url}/media/{s3_key}"
            log.info(f"[local] Stored {dest}, served at {url}")
            return url

        bucket = bucket or self.highlights_bucket

        self.client.upload_file(
            local_path,
            bucket,
            s3_key,
            ExtraArgs={
                "ContentType": content_type,
                "ACL": "public-read"
            }
        )

        url = (
            f"https://{bucket}.s3."
            f"{settings.aws_region}.amazonaws.com/"
            f"{s3_key}"
        )
        log.info(f"Uploaded to {url}")
        return url

    def generate_presigned_url(
        self,
        s3_key: str,
        bucket: str = None,
        expiry: int = 3600
    ) -> str:
        """Generate presigned URL for upload (S3 mode only)."""
        if not self.use_s3:
            raise RuntimeError(
                "generate_presigned_url() requires use_s3=True; "
                "local mode uses POST /api/v1/videos/upload instead"
            )
        bucket = bucket or self.raw_bucket
        url = self.client.generate_presigned_url(
            "put_object",
            Params={
                "Bucket": bucket,
                "Key": s3_key,
                "ContentType": "video/mp4"
            },
            ExpiresIn=expiry
        )
        return url
