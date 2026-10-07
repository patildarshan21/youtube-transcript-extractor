import pytest
import asyncio
from backend.utils import (
    validate_youtube_url,
    extract_video_id,
    format_timestamp,
    clean_caption_text,
    deduplicate_segments,
    get_device_and_compute_type,
    check_ffmpeg_installed,
)
from backend.youtube_service import get_instant_metadata, get_instant_subtitles
from backend.transcript_service import extract_full_transcript
from backend.whisper_service import whisper_manager, SUPPORTED_MODELS


def test_url_validation():
    # Valid YouTube URLs
    valid_cases = [
        "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        "https://youtu.be/jNQXAC9IVRw",
        "https://www.youtube.com/shorts/abcdefghijk",
        "https://www.youtube.com/embed/12345678901",
        "jNQXAC9IVRw",
    ]
    for url in valid_cases:
        valid, vid, canonical = validate_youtube_url(url)
        assert valid is True, f"Failed for {url}"
        assert vid is not None

    # Invalid URLs (Security: non-YouTube domains and arbitrary strings)
    invalid_cases = [
        "https://notyoutube.com/watch?v=dQw4w9WgXcQ",
        "https://youtube.com.attacker.com/watch?v=dQw4w9WgXcQ",
        "https://vimeo.com/12345678",
        "some random query text",
        "",
    ]
    for url in invalid_cases:
        valid, vid, canonical = validate_youtube_url(url)
        assert valid is False, f"Should have failed for {url}"


def test_timestamp_formatting():
    assert format_timestamp(0) == "00:00:00"
    assert format_timestamp(65) == "00:01:05"
    assert format_timestamp(3665) == "01:01:05"
    assert format_timestamp(12.345, include_ms=True) == "00:00:12.345"


def test_caption_cleaning():
    raw = "<c.colorE5E5E5>Hello &amp; welcome</c> to the <font color='#fff'>video</font>!\n\n"
    cleaned = clean_caption_text(raw)
    assert cleaned == "Hello & welcome to the video!"


def test_deduplication():
    segments = [
        {"start": 0.0, "end": 2.0, "text": "Hello world"},
        {"start": 2.0, "end": 4.0, "text": "Hello world"},  # exact duplicate
        {"start": 4.0, "end": 6.0, "text": "Next sentence"},
    ]
    deduped = deduplicate_segments(segments)
    assert len(deduped) == 2
    assert deduped[0]["text"] == "Hello world"
    assert deduped[0]["end"] == 4.0
    assert deduped[1]["text"] == "Next sentence"


def test_device_detection():
    device, compute_type, cuda_available = get_device_and_compute_type()
    assert device in ("cpu", "cuda")
    assert compute_type in ("int8", "float16", "float32")
    assert isinstance(cuda_available, bool)


def test_whisper_supported_models():
    assert "small" in SUPPORTED_MODELS
    assert "tiny" in SUPPORTED_MODELS
    assert "large-v3" in SUPPORTED_MODELS


def test_instant_metadata():
    meta = get_instant_metadata("jNQXAC9IVRw")
    assert meta is not None
    assert "Me at the zoo" in meta.get("title", "")
    assert meta.get("channel") == "jawed"


def test_real_video_transcript_extraction():
    # Test with standard public YouTube video ("Me at the zoo")
    res = asyncio.run(extract_full_transcript("https://www.youtube.com/watch?v=jNQXAC9IVRw", model_name="tiny"))
    assert res.success is True
    assert res.video_id == "jNQXAC9IVRw"
    assert "Me at the zoo" in res.title
    assert res.transcription_method in ("youtube_subtitles", "faster_whisper")
    assert len(res.transcript) > 0
    first_seg = res.transcript[0]
    assert first_seg.start >= 0.0
    assert first_seg.end > first_seg.start
    assert len(first_seg.text) > 0


def test_streamlit_app_url_switching():
    import os
    from streamlit.testing.v1 import AppTest
    app_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "app.py"))
    at = AppTest.from_file(app_path)
    at.run(timeout=30)

    # 1. First URL extraction
    at.text_input[0].input("jNQXAC9IVRw")
    at.button[0].click()
    at.run(timeout=30)
    assert at.session_state["transcript_data"] is not None
    assert at.session_state["transcript_data"]["video_id"] == "jNQXAC9IVRw"

    # 2. Second URL extraction - must immediately extract new video on first attempt
    at.text_input[0].input("dQw4w9WgXcQ")
    at.button[0].click()
    at.run(timeout=30)
    assert at.session_state["transcript_data"] is not None
    assert at.session_state["transcript_data"]["video_id"] == "dQw4w9WgXcQ"

