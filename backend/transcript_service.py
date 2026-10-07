import os
import uuid
import logging
import asyncio
from typing import Optional, Dict, Any, AsyncGenerator, Tuple

from backend.models import TranscriptResponse, TranscriptSegment
from backend.utils import (
    validate_youtube_url,
    format_timestamp,
    deduplicate_segments,
)
from backend.youtube_service import (
    get_instant_metadata,
    get_instant_subtitles,
    get_video_info,
    extract_youtube_subtitles,
    download_audio,
    YouTubeError,
    VideoUnavailableError,
    PrivateVideoError,
    AgeRestrictedError,
)
from backend.whisper_service import whisper_manager, DEFAULT_MODEL

logger = logging.getLogger("transcript_service")

# Base directory for temporary files
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEMP_DIR = os.path.join(PROJECT_ROOT, "temp")
os.makedirs(TEMP_DIR, exist_ok=True)

# High-speed in-memory LRU cache: (video_id, preferred_lang, model_name) -> TranscriptResponse
_TRANSCRIPT_CACHE: Dict[Tuple[str, str, str], TranscriptResponse] = {}
_CACHE_MAX_SIZE = 128


def get_cached_transcript(video_id: str, preferred_lang: str, model_name: str) -> Optional[TranscriptResponse]:
    """Retrieve already extracted transcript instantly in 0ms."""
    key = (video_id, preferred_lang.lower(), model_name.lower())
    return _TRANSCRIPT_CACHE.get(key)


def set_cached_transcript(video_id: str, preferred_lang: str, model_name: str, response: TranscriptResponse) -> None:
    """Cache transcript in memory."""
    if len(_TRANSCRIPT_CACHE) >= _CACHE_MAX_SIZE:
        # Remove oldest entry
        first_key = next(iter(_TRANSCRIPT_CACHE))
        _TRANSCRIPT_CACHE.pop(first_key, None)
    key = (video_id, preferred_lang.lower(), model_name.lower())
    _TRANSCRIPT_CACHE[key] = response


async def extract_full_transcript(
    url: str,
    model_name: str = DEFAULT_MODEL,
    preferred_lang: str = "en",
) -> TranscriptResponse:
    """
    Standard transcript extraction reusing the unified multi-tier streaming workflow.
    Guarantees consistent caching, deduplication, and temporary file cleanup.
    """
    is_valid, video_id, canonical_url = validate_youtube_url(url)
    if not is_valid or not canonical_url or not video_id:
        raise VideoUnavailableError("Invalid YouTube URL or video ID. Please enter a valid YouTube link.")

    last_error: Optional[str] = None
    async for event in stream_transcript_progress(url, model_name, preferred_lang):
        stage = event.get("stage")
        if stage == "complete" and "data" in event:
            return TranscriptResponse(**event["data"])
        elif stage == "error":
            last_error = event.get("message")

    raise YouTubeError(last_error or "Transcript extraction failed.")


async def stream_transcript_progress(
    url: str,
    model_name: str = DEFAULT_MODEL,
    preferred_lang: str = "en",
) -> AsyncGenerator[Dict[str, Any], None]:
    """
    Streaming generator for real-time progress updates and transcript generation.
    Emits instant updates without synthetic pauses.
    """
    yield {
        "stage": "validating",
        "message": "Validating YouTube URL...",
        "percent": 10,
    }

    is_valid, video_id, canonical_url = validate_youtube_url(url)
    if not is_valid or not canonical_url or not video_id:
        yield {
            "stage": "error",
            "message": "Invalid YouTube URL or video ID. Please check the URL and try again.",
            "percent": 0,
        }
        return

    # Check cache for instantaneous return
    cached = get_cached_transcript(video_id, preferred_lang, model_name)
    if cached:
        yield {
            "stage": "complete",
            "message": "Instant transcript retrieved from cache!",
            "percent": 100,
            "data": cached.model_dump(),
        }
        return

    yield {
        "stage": "fetching_info",
        "message": "Connecting to YouTube fast-path...",
        "percent": 25,
    }

    loop = asyncio.get_running_loop()

    # Tier 1: Instant Fast-Path
    instant_meta_task = loop.run_in_executor(None, get_instant_metadata, video_id)
    instant_sub_task = loop.run_in_executor(None, get_instant_subtitles, video_id, preferred_lang)

    instant_meta, instant_sub = await asyncio.gather(instant_meta_task, instant_sub_task)

    if instant_sub:
        yield {
            "stage": "extracting_subtitles",
            "message": f"Instant subtitles extracted ({instant_sub[1]})!",
            "percent": 85,
        }

        cleaned_segments, detected_lang, method, estimated_duration = instant_sub
        title = (instant_meta.get("title") if instant_meta else None) or "YouTube Video"
        channel = (instant_meta.get("channel") if instant_meta else None)
        thumbnail = (instant_meta.get("thumbnail") if instant_meta else None) or f"https://i.ytimg.com/vi/{video_id}/hqdefault.jpg"

        transcript_objs = [
            TranscriptSegment(start=seg["start"], end=seg["end"], text=seg["text"])
            for seg in cleaned_segments
        ]
        total_words = sum(len(seg.text.split()) for seg in transcript_objs)

        response_data = TranscriptResponse(
            success=True,
            video_id=video_id,
            video_url=canonical_url,
            title=title,
            channel=channel,
            language=detected_lang,
            duration=estimated_duration,
            duration_formatted=format_timestamp(estimated_duration),
            thumbnail=thumbnail,
            method="youtube_subtitles",
            transcription_method="youtube_subtitles",
            model_used=None,
            transcript=transcript_objs,
            word_count=total_words,
            segment_count=len(transcript_objs),
        )
        set_cached_transcript(video_id, preferred_lang, model_name, response_data)

        yield {
            "stage": "complete",
            "message": "Transcript extraction complete!",
            "percent": 100,
            "data": response_data.model_dump(),
        }
        return

    # Tier 2: Full yt-dlp Subtitle Strategy
    yield {
        "stage": "checking_subtitles",
        "message": "Inspecting video formats and subtitle streams with yt-dlp...",
        "percent": 35,
    }

    try:
        info = await loop.run_in_executor(None, get_video_info, canonical_url)
    except (VideoUnavailableError, PrivateVideoError, AgeRestrictedError, YouTubeError) as e:
        yield {
            "stage": "error",
            "message": str(e),
            "percent": 0,
        }
        return
    except Exception as e:
        logger.error(f"Unexpected error retrieving info: {e}", exc_info=True)
        yield {
            "stage": "error",
            "message": "Failed to connect to YouTube. Please check network connection.",
            "percent": 0,
        }
        return

    title = info.get("title", "Untitled Video")
    channel = info.get("uploader") or info.get("channel")
    duration = float(info.get("duration") or 0.0)
    thumbnail = info.get("thumbnail")

    sub_result = await loop.run_in_executor(None, extract_youtube_subtitles, info, preferred_lang)

    temp_audio_path: Optional[str] = None
    transcription_method = "youtube_subtitles"
    model_used = None

    try:
        if sub_result:
            segments_raw, detected_lang, method = sub_result
            transcription_method = "youtube_subtitles"
            detected_language = detected_lang

            yield {
                "stage": "extracting_subtitles",
                "message": f"Subtitles found ({detected_lang})! Extracting full transcript...",
                "percent": 80,
            }
            cleaned_segments = deduplicate_segments(segments_raw)
        else:
            # Tier 3: Fallback to faster-whisper
            transcription_method = "faster_whisper"
            model_used = model_name

            yield {
                "stage": "downloading_audio",
                "message": "No subtitles found. Downloading audio stream with yt-dlp...",
                "percent": 45,
            }

            req_temp_dir = os.path.join(TEMP_DIR, f"stream_{uuid.uuid4().hex[:8]}")
            os.makedirs(req_temp_dir, exist_ok=True)

            progress_queue: asyncio.Queue = asyncio.Queue()

            def audio_progress(pct: int, msg: str):
                scaled_pct = 45 + int(pct * 0.15)
                loop.call_soon_threadsafe(
                    progress_queue.put_nowait,
                    {"stage": "downloading_audio", "message": msg, "percent": scaled_pct}
                )

            download_task = loop.run_in_executor(
                None, download_audio, canonical_url, req_temp_dir, audio_progress
            )

            while not download_task.done():
                try:
                    update = await asyncio.wait_for(progress_queue.get(), timeout=0.1)
                    yield update
                except asyncio.TimeoutError:
                    pass

            temp_audio_path = await download_task

            yield {
                "stage": "loading_model",
                "message": f"Loading Whisper model ('{model_name}')...",
                "percent": 65,
            }

            def whisper_progress(pct: int, msg: str):
                loop.call_soon_threadsafe(
                    progress_queue.put_nowait,
                    {"stage": "transcribing", "message": msg, "percent": pct}
                )

            transcribe_task = loop.run_in_executor(
                None, whisper_manager.transcribe, temp_audio_path, model_name, duration, whisper_progress
            )

            while not transcribe_task.done():
                try:
                    update = await asyncio.wait_for(progress_queue.get(), timeout=0.1)
                    yield update
                except asyncio.TimeoutError:
                    pass

            cleaned_segments, detected_language = await transcribe_task

        yield {
            "stage": "processing",
            "message": "Formatting complete transcript...",
            "percent": 98,
        }

        transcript_objs = [
            TranscriptSegment(start=seg["start"], end=seg["end"], text=seg["text"])
            for seg in cleaned_segments
        ]
        total_words = sum(len(seg.text.split()) for seg in transcript_objs)

        response_data = TranscriptResponse(
            success=True,
            video_id=video_id,
            video_url=canonical_url,
            title=title,
            channel=channel,
            language=detected_language,
            duration=duration,
            duration_formatted=format_timestamp(duration),
            thumbnail=thumbnail,
            method=transcription_method,
            transcription_method=transcription_method,
            model_used=model_used,
            transcript=transcript_objs,
            word_count=total_words,
            segment_count=len(transcript_objs),
        )
        set_cached_transcript(video_id, preferred_lang, model_name, response_data)

        yield {
            "stage": "complete",
            "message": "Transcript extraction complete!",
            "percent": 100,
            "data": response_data.model_dump(),
        }

    except (VideoUnavailableError, PrivateVideoError, AgeRestrictedError, YouTubeError) as e:
        yield {
            "stage": "error",
            "message": str(e),
            "percent": 0,
        }
    except Exception as e:
        logger.error(f"Unexpected error in stream: {e}", exc_info=True)
        yield {
            "stage": "error",
            "message": f"Transcription error: {str(e)}",
            "percent": 0,
        }
    finally:
        if temp_audio_path and os.path.exists(temp_audio_path):
            try:
                os.remove(temp_audio_path)
                parent_dir = os.path.dirname(temp_audio_path)
                if os.path.exists(parent_dir) and parent_dir != TEMP_DIR:
                    os.rmdir(parent_dir)
            except Exception as e:
                logger.warning(f"Error removing temporary audio: {e}")
