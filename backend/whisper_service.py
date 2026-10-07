import os
import gc
import logging
import threading
from typing import Optional, Tuple, List, Dict, Any, Callable

# Disable Hugging Face symlink warning on Windows systems
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# Compatibility patch for PyAV (av) metadata_errors parameter in newer versions
try:
    import av
    _orig_av_open = av.open

    def _safe_av_open(*args, **kwargs):
        kwargs.pop("metadata_errors", None)
        return _orig_av_open(*args, **kwargs)

    av.open = _safe_av_open
except Exception:
    pass

from faster_whisper import WhisperModel
from backend.utils import get_device_and_compute_type, clean_caption_text, deduplicate_segments

logger = logging.getLogger("whisper_service")

SUPPORTED_MODELS = ["tiny", "base", "small", "medium", "large-v3"]
DEFAULT_MODEL = "small"


class WhisperManager:
    """
    Thread-safe Singleton manager for caching and reusing faster-whisper models.
    Automatically handles CUDA vs CPU device configuration and prevents redundant
    re-loading of weights across requests.
    """
    _instance: Optional["WhisperManager"] = None
    _singleton_lock = threading.Lock()

    def __new__(cls) -> "WhisperManager":
        with cls._singleton_lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._init_manager()
            return cls._instance

    def _init_manager(self) -> None:
        self.device, self.compute_type, self.cuda_available = get_device_and_compute_type()
        self.current_model: Optional[WhisperModel] = None
        self.current_model_name: Optional[str] = None
        self.model_lock = threading.Lock()
        logger.info(
            f"Initialized WhisperManager on device='{self.device}', "
            f"compute_type='{self.compute_type}', cuda_available={self.cuda_available}"
        )

    def get_model(self, model_name: str = DEFAULT_MODEL) -> WhisperModel:
        """
        Retrieve or load the requested Whisper model.
        Reuses cached model if the requested model name matches the active one.
        """
        model_name = model_name.strip().lower()
        if model_name not in SUPPORTED_MODELS:
            logger.warning(f"Requested model '{model_name}' not in supported list. Falling back to '{DEFAULT_MODEL}'.")
            model_name = DEFAULT_MODEL

        with self.model_lock:
            if self.current_model is not None and self.current_model_name == model_name:
                logger.info(f"Reusing already loaded Whisper model '{model_name}'")
                return self.current_model

            logger.info(
                f"Loading faster-whisper model '{model_name}' on {self.device} "
                f"with compute_type='{self.compute_type}'..."
            )

            # Explicitly free previous model resources
            if self.current_model is not None:
                del self.current_model
                self.current_model = None
                gc.collect()

            try:
                # Load Whisper model with configured device and precision
                model = WhisperModel(
                    model_name,
                    device=self.device,
                    compute_type=self.compute_type,
                    cpu_threads=max(1, os.cpu_count() or 4) if self.device == "cpu" else 4,
                )
                self.current_model = model
                self.current_model_name = model_name
                logger.info(f"Model '{model_name}' loaded successfully.")
                return self.current_model
            except Exception as e:
                logger.error(f"Failed to load Whisper model '{model_name}': {e}", exc_info=True)
                # If loading on cuda failed or compute type failed, fallback to cpu int8
                if self.device == "cuda" or self.compute_type != "int8":
                    logger.warning("Attempting CPU int8 fallback for WhisperModel...")
                    self.device = "cpu"
                    self.compute_type = "int8"
                    model = WhisperModel(
                        model_name,
                        device="cpu",
                        compute_type="int8",
                        cpu_threads=max(1, os.cpu_count() or 4),
                    )
                    self.current_model = model
                    self.current_model_name = model_name
                    return self.current_model
                raise

    def transcribe(
        self,
        audio_path: str,
        model_name: str = DEFAULT_MODEL,
        duration: float = 0.0,
        progress_callback: Optional[Callable[[int, str], None]] = None
    ) -> Tuple[List[Dict[str, Any]], str]:
        """
        Transcribe the audio file using faster-whisper.
        Processes segments incrementally to prevent memory spikes on long recordings
        and reports real-time percentage progress.
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        if progress_callback:
            progress_callback(5, f"Loading Whisper model '{model_name}'...")

        model = self.get_model(model_name)

        if progress_callback:
            progress_callback(10, "Transcribing audio stream with faster-whisper...")

        logger.info(f"Starting transcription for {audio_path} using model '{model_name}'")

        # Transcribe with VAD filter to reduce silence hallucination
        segments_generator, info = model.transcribe(
            audio_path,
            beam_size=5,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500),
            word_timestamps=False,
        )

        detected_language = info.language or "en"
        logger.info(f"Detected spoken language: '{detected_language}' (probability: {info.language_probability:.2f})")

        raw_segments: List[Dict[str, Any]] = []
        effective_duration = duration if duration > 0 else (info.duration or 0.0)

        for segment in segments_generator:
            text = clean_caption_text(segment.text)
            if text:
                raw_segments.append({
                    "start": round(segment.start, 2),
                    "end": round(segment.end, 2),
                    "text": text,
                })

            # Calculate and report incremental progress
            if progress_callback and effective_duration > 0:
                pct = int((segment.end / effective_duration) * 85) + 10
                pct = min(95, max(10, pct))
                progress_callback(pct, f"Transcribing audio: {pct}% complete...")

        # Deduplicate and normalize segments
        cleaned_segments = deduplicate_segments(raw_segments)
        logger.info(f"Transcription complete: produced {len(cleaned_segments)} segments")

        return cleaned_segments, detected_language


# Module-level singleton instance
whisper_manager = WhisperManager()
