from datetime import datetime, timedelta


def seconds_to_timestamp(seconds: float) -> str:
    """Convert seconds to HH:MM:SS.mmm format"""
    td = timedelta(seconds=max(0.0, seconds))
    total_seconds = int(td.total_seconds())
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    millis = int((seconds - int(seconds)) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"


def timestamp_to_seconds(timestamp: str) -> float:
    """Convert HH:MM:SS.mmm format to seconds"""
    parts = timestamp.split(":")
    hours, minutes = int(parts[0]), int(parts[1])
    seconds = float(parts[2])
    return hours * 3600 + minutes * 60 + seconds


def format_duration(seconds: float) -> str:
    """Human readable duration, e.g. '5m 32s'"""
    minutes, secs = divmod(int(seconds), 60)
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def utc_now() -> datetime:
    return datetime.utcnow()
