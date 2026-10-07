from typing import List, Optional, Any, Dict
from pydantic import BaseModel, Field


class TranscriptRequest(BaseModel):
    url: str = Field(..., description="YouTube video URL or video ID")
    model: str = Field(default="small", description="Whisper model name: tiny, base, small, medium, large-v3")
    preferred_language: Optional[str] = Field(default="en", description="Preferred subtitle language code")


class TranscriptSegment(BaseModel):
    start: float = Field(..., description="Start timestamp in seconds")
    end: float = Field(..., description="End timestamp in seconds")
    text: str = Field(..., description="Spoken transcription text")


class TranscriptResponse(BaseModel):
    success: bool = True
    video_id: str
    video_url: str
    title: str
    channel: Optional[str] = None
    language: str = "en"
    duration: float = 0.0
    duration_formatted: str = "00:00:00"
    thumbnail: Optional[str] = None
    method: str = "youtube_subtitles"  # "youtube_subtitles" or "faster_whisper"
    transcription_method: str = "youtube_subtitles"  # exact match for JSON download schema
    model_used: Optional[str] = None
    transcript: List[TranscriptSegment]
    word_count: int = 0
    segment_count: int = 0


class HealthResponse(BaseModel):
    status: str = "healthy"
    device: str
    compute_type: str
    cuda_available: bool
    ffmpeg_available: bool
    supported_models: List[str]
    cached_model: Optional[str] = None
