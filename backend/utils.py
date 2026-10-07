import re
import html
import shutil
import urllib.parse
from typing import Optional, Tuple, List, Dict, Any


# Regular expressions for YouTube URL matching and ID extraction
YOUTUBE_DOMAINS = {
    "youtube.com",
    "www.youtube.com",
    "m.youtube.com",
    "music.youtube.com",
    "youtu.be",
    "www.youtu.be"
}

VIDEO_ID_PATTERN = re.compile(r"^[a-zA-Z0-9_-]{11}$")

URL_PATTERNS = [
    # standard watch: https://www.youtube.com/watch?v=VIDEO_ID
    re.compile(r"(?:https?://)?(?:www\.|m\.|music\.)?youtube\.com/watch\?.*v=([a-zA-Z0-9_-]{11})"),
    # short url: https://youtu.be/VIDEO_ID
    re.compile(r"(?:https?://)?(?:www\.)?youtu\.be/([a-zA-Z0-9_-]{11})"),
    # embed: https://www.youtube.com/embed/VIDEO_ID
    re.compile(r"(?:https?://)?(?:www\.)?youtube\.com/embed/([a-zA-Z0-9_-]{11})"),
    # shorts: https://www.youtube.com/shorts/VIDEO_ID
    re.compile(r"(?:https?://)?(?:www\.)?youtube\.com/shorts/([a-zA-Z0-9_-]{11})"),
    # live: https://www.youtube.com/live/VIDEO_ID
    re.compile(r"(?:https?://)?(?:www\.)?youtube\.com/live/([a-zA-Z0-9_-]{11})"),
]

# Regex for stripping HTML/WebVTT formatting tags (e.g., <c>, <00:00:01.000>, <font>)
TAG_PATTERN = re.compile(r"<[^>]+>")


def extract_video_id(url_or_id: str) -> Optional[str]:
    """
    Extract an 11-character YouTube video ID from a URL or raw ID string.
    Strictly ensures the domain is youtube.com, youtu.be, or a direct 11-char ID.
    """
    if not url_or_id:
        return None

    candidate = url_or_id.strip()

    # Check if candidate is already a raw 11-character video ID (no slashes, no dots)
    if VIDEO_ID_PATTERN.match(candidate):
        return candidate

    # Parse URL
    try:
        # Add https:// if missing a scheme
        if not candidate.startswith(("http://", "https://")):
            parsed_test = urllib.parse.urlparse("https://" + candidate)
        else:
            parsed_test = urllib.parse.urlparse(candidate)

        netloc = parsed_test.netloc.lower()
        # Remove port if present
        if ":" in netloc:
            netloc = netloc.split(":")[0]

        # Strictly check allowed domains
        is_allowed_domain = False
        for allowed in YOUTUBE_DOMAINS:
            if netloc == allowed or netloc.endswith("." + allowed):
                is_allowed_domain = True
                break

        if not is_allowed_domain:
            return None

        # Check query parameter 'v'
        query_params = urllib.parse.parse_qs(parsed_test.query)
        if "v" in query_params and query_params["v"]:
            vid = query_params["v"][0]
            if VIDEO_ID_PATTERN.match(vid):
                return vid

        # Check path parts
        path_parts = [p for p in parsed_test.path.split("/") if p]
        if "youtu.be" in netloc and path_parts:
            # e.g., youtu.be/VIDEO_ID
            cand_id = path_parts[0]
            if VIDEO_ID_PATTERN.match(cand_id):
                return cand_id

        if len(path_parts) >= 2 and path_parts[0] in ("shorts", "embed", "live", "v"):
            # e.g., youtube.com/shorts/VIDEO_ID
            cand_id = path_parts[1]
            if VIDEO_ID_PATTERN.match(cand_id):
                return cand_id

        # Direct /watch?v= in URL path or regex fallback on sanitized string
        for pattern in URL_PATTERNS:
            match = pattern.search(candidate)
            if match:
                vid = match.group(1)
                if VIDEO_ID_PATTERN.match(vid):
                    return vid
    except Exception:
        pass

    return None


def validate_youtube_url(url: str) -> Tuple[bool, Optional[str], Optional[str]]:
    """
    Validate that the provided string is a valid YouTube URL or video ID.
    Returns:
        (is_valid, video_id, canonical_url)
    """
    video_id = extract_video_id(url)
    if not video_id:
        return False, None, None

    canonical_url = f"https://www.youtube.com/watch?v={video_id}"
    return True, video_id, canonical_url


def format_timestamp(seconds: float, include_ms: bool = False) -> str:
    """
    Convert seconds (float) into formatted string [HH:MM:SS] or [HH:MM:SS.mmm].
    Example: 125.45 -> "00:02:05"
    """
    if seconds is None or seconds < 0:
        seconds = 0.0

    total_seconds = int(seconds)
    ms = int(round((seconds - total_seconds) * 1000))

    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60

    if include_ms:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}.{ms:03d}"
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def clean_caption_text(text: str) -> str:
    """
    Clean caption/transcript text:
    - Strips HTML and WebVTT timing tags (<c>, <00:00:01.000>, etc.)
    - Unescapes HTML entities (&amp;, &#39;, &quot;, &lt;, &gt;)
    - Removes zero-width and non-breaking spaces
    - Collapses repeated whitespace and newline characters
    - Fixes spacing before punctuation
    """
    if not text:
        return ""

    # Unescape HTML entities
    cleaned = html.unescape(text)

    # Remove tags like <c.colorE5E5E5>, </c>, <00:00:00.000>, <font ...>
    cleaned = TAG_PATTERN.sub("", cleaned)

    # Normalize various Unicode whitespace characters
    cleaned = cleaned.replace("\xa0", " ").replace("\u200b", "")

    # Replace newlines with spaces
    cleaned = re.sub(r"[\r\n]+", " ", cleaned)

    # Collapse multiple spaces
    cleaned = re.sub(r"\s+", " ", cleaned)

    # Remove unwanted space before common punctuation marks
    cleaned = re.sub(r"\s+([.,!?;:])", r"\1", cleaned).strip()

    return cleaned


def deduplicate_segments(segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Clean and deduplicate segments from auto-generated captions or Whisper output.
    Handles:
    - Empty or whitespace-only texts
    - Exact duplicate lines
    - Auto-caption sliding window overlaps (where sentence prefixes repeat)
    - Ensures start <= end
    """
    if not segments:
        return []

    cleaned_segments: List[Dict[str, Any]] = []

    for seg in segments:
        raw_text = seg.get("text", "")
        text = clean_caption_text(raw_text)
        if not text:
            continue

        start = round(float(seg.get("start", 0.0)), 2)
        end = round(float(seg.get("end", start + 0.1)), 2)
        if end <= start:
            end = round(start + 1.0, 2)

        # Check against the previous segment
        if cleaned_segments:
            prev = cleaned_segments[-1]
            prev_text = prev["text"]

            # Case 1: Exact duplicate text
            if text == prev_text:
                # Extend end timestamp of previous segment if this one ends later
                if end > prev["end"]:
                    prev["end"] = end
                continue

            # Case 2: Current text starts with previous text (common in auto captions)
            # e.g., prev: "hello", current: "hello and welcome"
            if text.startswith(prev_text) and len(text) > len(prev_text):
                # Replace previous text or extract the new portion
                new_part = text[len(prev_text):].strip()
                if new_part:
                    cleaned_segments.append({
                        "start": prev["end"],
                        "end": end,
                        "text": new_part
                    })
                continue

            # Case 3: Previous text starts with current text (duplicate snippet)
            if prev_text.startswith(text):
                continue

            # Case 4: Overlapping words between end of prev and start of current
            # Check for word-level suffix/prefix overlap
            prev_words = prev_text.split()
            curr_words = text.split()
            max_overlap = min(len(prev_words), len(curr_words), 5)
            overlap_found = False

            for k in range(max_overlap, 1, -1):
                if prev_words[-k:] == curr_words[:k]:
                    # The first k words of current are identical to last k words of previous
                    non_overlapping = " ".join(curr_words[k:]).strip()
                    if non_overlapping:
                        cleaned_segments.append({
                            "start": start,
                            "end": end,
                            "text": non_overlapping
                        })
                    overlap_found = True
                    break

            if overlap_found:
                continue

        cleaned_segments.append({
            "start": start,
            "end": end,
            "text": text
        })

    return cleaned_segments


def parse_vtt_content(vtt_text: str) -> List[Dict[str, Any]]:
    """
    Parse WebVTT file content into a list of transcript segment dicts.
    """
    segments: List[Dict[str, Any]] = []
    lines = vtt_text.splitlines()

    time_pattern = re.compile(
        r"(\d{1,2}:)?(\d{2}):(\d{2})[.,](\d{3})\s*-->\s*(\d{1,2}:)?(\d{2}):(\d{2})[.,](\d{3})"
    )

    def parse_time_to_seconds(h: Optional[str], m: str, s: str, ms: str) -> float:
        hours = int(h.rstrip(":")) if h else 0
        minutes = int(m)
        secs = int(s)
        millis = int(ms)
        return round(hours * 3600 + minutes * 60 + secs + millis / 1000.0, 3)

    idx = 0
    total = len(lines)
    while idx < total:
        line = lines[idx].strip()
        match = time_pattern.search(line)
        if match:
            start_sec = parse_time_to_seconds(match.group(1), match.group(2), match.group(3), match.group(4))
            end_sec = parse_time_to_seconds(match.group(5), match.group(6), match.group(7), match.group(8))

            text_lines = []
            idx += 1
            while idx < total and lines[idx].strip() and not time_pattern.search(lines[idx]):
                text_lines.append(lines[idx].strip())
                idx += 1

            combined_text = " ".join(text_lines)
            cleaned = clean_caption_text(combined_text)
            if cleaned:
                segments.append({
                    "start": start_sec,
                    "end": end_sec,
                    "text": cleaned
                })
        else:
            idx += 1

    return deduplicate_segments(segments)


def parse_json3_content(json3_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Parse YouTube timedtext JSON3 subtitle format into segment dicts.
    """
    segments: List[Dict[str, Any]] = []
    events = json3_data.get("events", [])

    for event in events:
        if "tStartMs" not in event:
            continue

        start_ms = event.get("tStartMs", 0)
        dur_ms = event.get("dDurationMs", 1000)
        start_sec = round(start_ms / 1000.0, 2)
        end_sec = round((start_ms + dur_ms) / 1000.0, 2)

        segs = event.get("segs", [])
        text_parts = []
        for seg in segs:
            chunk = seg.get("utf8", "")
            if chunk:
                text_parts.append(chunk)

        combined_text = clean_caption_text("".join(text_parts))
        if combined_text:
            segments.append({
                "start": start_sec,
                "end": end_sec,
                "text": combined_text
            })

    return deduplicate_segments(segments)


def get_device_and_compute_type() -> Tuple[str, str, bool]:
    """
    Detect CUDA / GPU availability safely without crashing if CUDA is missing.
    Returns:
        (device, compute_type, cuda_available)
    """
    cuda_available = False
    try:
        import torch
        import ctranslate2
        if torch.cuda.is_available() and ctranslate2.get_cuda_device_count() > 0:
            cuda_available = True
    except Exception:
        cuda_available = False

    if cuda_available:
        return "cuda", "float16", True
    else:
        return "cpu", "int8", False


def check_ffmpeg_installed() -> bool:
    """
    Check if FFmpeg executable is available in PATH.
    """
    return shutil.which("ffmpeg") is not None
