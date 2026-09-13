import concurrent.futures
import numpy as np
import librosa
from typing import List, Dict
from app.config import settings
from app.utils.logger import log

# A 38-second clip hung inside librosa.load() itself for 8+ minutes on
# Render's free tier without erroring -- root cause unconfirmed (likely
# a fallback audio decoder backend misbehaving in that container), but
# whatever it is, no single video should be able to wedge the whole
# job forever. Bound it with a hard timeout and degrade to visual-only
# detection rather than hang.
AUDIO_ANALYSIS_TIMEOUT_SECONDS = 45


class AudioAnalyzer:
    """
    Analyzes cricket match audio to detect highlight moments.
    Cricket has very distinct audio patterns:
    - SIX/FOUR -> sudden loud crowd roar
    - WICKET -> excited crowd + commentary spike
    - CATCH -> short sharp crowd reaction
    - Normal play -> moderate background crowd noise
    """

    def __init__(self):
        self.spike_threshold_multiplier = (
            settings.audio_spike_threshold
        )
        self.min_highlight_gap = 5.0  # seconds between highlights
        self.context_window = 3.0     # seconds before spike

    def analyze(
        self,
        audio_path: str
    ) -> List[Dict]:
        """
        Main analysis function, with a hard timeout. If audio analysis
        hangs or fails, callers still get a (possibly empty) list back
        -- the pipeline treats this as one signal among several
        (combined with YOLO visual detections), so degrading to
        visual-only is a safe fallback rather than blocking the whole
        video.
        """
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        future = executor.submit(self._analyze_impl, audio_path)
        try:
            return future.result(timeout=AUDIO_ANALYSIS_TIMEOUT_SECONDS)
        except concurrent.futures.TimeoutError:
            log.error(
                f"Audio analysis timed out after "
                f"{AUDIO_ANALYSIS_TIMEOUT_SECONDS}s on {audio_path} -- "
                f"continuing with visual-only detection. (The stuck "
                f"worker thread is abandoned, not killed -- Python "
                f"can't force-terminate a thread -- but the job moves "
                f"on instead of hanging.)"
            )
            return []
        except Exception as e:
            log.error(f"Audio analysis failed: {str(e)}")
            return []
        finally:
            # wait=False: don't block here waiting for a possibly
            # still-hung worker thread to finish.
            executor.shutdown(wait=False)

    def _analyze_impl(
        self,
        audio_path: str
    ) -> List[Dict]:
        """The actual analysis, run on a worker thread by analyze()."""
        log.info(f"Analyzing audio: {audio_path}")

        try:
            # Load audio
            audio, sample_rate = librosa.load(
                audio_path,
                sr=22050,    # Standard sample rate
                mono=True    # Convert to mono
            )
            log.info("librosa.load() returned")

            duration = len(audio) / sample_rate
            log.info(
                f"Audio duration: {duration:.1f}s, "
                f"Sample rate: {sample_rate}"
            )

            # Run all analysis methods
            volume_spikes = self._detect_volume_spikes(
                audio, sample_rate
            )
            spectral_spikes = self._detect_spectral_changes(
                audio, sample_rate
            )
            crowd_excitement = self._detect_crowd_excitement(
                audio, sample_rate
            )

            # Combine all detections
            all_detections = self._combine_detections(
                volume_spikes,
                spectral_spikes,
                crowd_excitement,
                duration=duration
            )

            # Remove duplicates within 5 seconds
            filtered = self._deduplicate(
                all_detections,
                self.min_highlight_gap
            )

            log.info(
                f"Audio analysis found "
                f"{len(filtered)} potential highlights"
            )
            return filtered

        except Exception as e:
            log.error(f"Audio analysis failed: {str(e)}")
            return []

    def _detect_volume_spikes(
        self,
        audio: np.ndarray,
        sr: int
    ) -> List[Dict]:
        """Detect sudden volume increases (crowd roar)"""
        frame_length = int(sr * 0.5)
        hop_length = int(sr * 0.25)

        rms = librosa.feature.rms(
            y=audio,
            frame_length=frame_length,
            hop_length=hop_length
        )[0]

        times = librosa.frames_to_time(
            np.arange(len(rms)),
            sr=sr,
            hop_length=hop_length
        )

        mean_rms = np.mean(rms)
        std_rms = np.std(rms)
        threshold = (
            mean_rms +
            self.spike_threshold_multiplier * std_rms
        )

        spike_indices = np.where(rms > threshold)[0]

        detections = []
        for idx in spike_indices:
            timestamp = float(times[idx])
            highlight_time = max(
                0, timestamp - self.context_window
            )

            detections.append({
                "timestamp": highlight_time,
                "peak_timestamp": timestamp,
                "confidence": float(
                    min(1.0, (rms[idx] - threshold) /
                    (mean_rms * 2 + 1e-6) + 0.6)
                ),
                "method": "volume_spike",
                "magnitude": float(rms[idx]),
                "threshold": float(threshold),
            })

        log.debug(
            f"Volume spikes found: {len(detections)}"
        )
        return detections

    def _detect_spectral_changes(
        self,
        audio: np.ndarray,
        sr: int
    ) -> List[Dict]:
        """
        Detect sudden frequency changes.
        Crowd roar has specific spectral signature.
        """
        hop_length = 512
        spectral_flux = librosa.onset.onset_strength(
            y=audio,
            sr=sr,
            hop_length=hop_length
        )

        times = librosa.frames_to_time(
            np.arange(len(spectral_flux)),
            sr=sr,
            hop_length=hop_length
        )

        peaks = librosa.util.peak_pick(
            spectral_flux,
            pre_max=3,
            post_max=3,
            pre_avg=10,
            post_avg=10,
            delta=0.5,
            wait=10
        )

        detections = []
        for peak_idx in peaks:
            timestamp = float(times[peak_idx])
            highlight_time = max(
                0, timestamp - self.context_window
            )

            detections.append({
                "timestamp": highlight_time,
                "peak_timestamp": timestamp,
                "confidence": float(
                    min(0.8,
                        spectral_flux[peak_idx] / 10.0)
                ),
                "method": "spectral_change",
                "magnitude": float(spectral_flux[peak_idx]),
            })

        log.debug(
            f"Spectral changes found: {len(detections)}"
        )
        return detections

    def _detect_crowd_excitement(
        self,
        audio: np.ndarray,
        sr: int
    ) -> List[Dict]:
        """
        Detect crowd excitement using spectral centroid.
        Excited crowd has higher frequency content.
        """
        hop_length = int(sr * 0.5)

        centroid = librosa.feature.spectral_centroid(
            y=audio,
            sr=sr,
            hop_length=hop_length
        )[0]

        zcr = librosa.feature.zero_crossing_rate(
            audio,
            hop_length=hop_length
        )[0]

        times = librosa.frames_to_time(
            np.arange(len(centroid)),
            sr=sr,
            hop_length=hop_length
        )

        centroid_norm = (
            centroid - np.min(centroid)
        ) / (np.max(centroid) - np.min(centroid) + 1e-6)

        zcr_norm = (
            zcr - np.min(zcr)
        ) / (np.max(zcr) - np.min(zcr) + 1e-6)

        excitement = (centroid_norm + zcr_norm) / 2
        threshold = np.percentile(excitement, 85)

        high_excitement = np.where(
            excitement > threshold
        )[0]

        detections = []
        for idx in high_excitement:
            timestamp = float(times[idx])
            highlight_time = max(
                0, timestamp - self.context_window
            )

            detections.append({
                "timestamp": highlight_time,
                "peak_timestamp": timestamp,
                "confidence": float(excitement[idx] * 0.7),
                "method": "crowd_excitement",
                "magnitude": float(excitement[idx]),
            })

        return detections

    def _combine_detections(
        self,
        *detection_lists,
        duration: float
    ) -> List[Dict]:
        """Combine all detection methods"""
        all_detections = []
        for dl in detection_lists:
            all_detections.extend(dl)

        all_detections.sort(key=lambda x: x["timestamp"])

        return all_detections

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
