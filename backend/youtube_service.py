import os
import shutil
import logging
import urllib.request
import json
from typing import Optional, Tuple, List, Dict, Any, Callable
import yt_dlp

try:
    from youtube_transcript_api import YouTubeTranscriptApi
except ImportError:
    YouTubeTranscriptApi = None

from backend.utils import (
    validate_youtube_url,
    format_timestamp,
    parse_json3_content,
    parse_vtt_content,
    clean_caption_text,
    deduplicate_segments,
)

logger = logging.getLogger("youtube_service")


class YouTubeError(Exception):
    """Base exception for YouTube operations."""
    pass


class VideoUnavailableError(YouTubeError):
    """Raised when a video is unavailable, deleted, or ID is invalid."""
    pass


class PrivateVideoError(YouTubeError):
    """Raised when a video is private or hidden."""
    pass


class AgeRestrictedError(YouTubeError):
    """Raised when a video is age-restricted and requires sign-in."""
    pass


def get_instant_metadata(video_id: str) -> Optional[Dict[str, Any]]:
    """
    Fetch basic video metadata (title, author, thumbnail) via YouTube oEmbed
    in under 200 milliseconds without running yt-dlp.
    """
    try:
        url = f"https://www.youtube.com/oembed?url=https://www.youtube.com/watch?v={video_id}&format=json"
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
        )
        with urllib.request.urlopen(req, timeout=3) as resp:
            data = json.loads(resp.read().decode())
            return {
                "title": data.get("title", "YouTube Video"),
                "channel": data.get("author_name"),
                "thumbnail": data.get("thumbnail_url") or f"https://i.ytimg.com/vi/{video_id}/hqdefault.jpg",
            }
    except Exception as e:
        logger.debug(f"Instant oEmbed metadata fetch skipped: {e}")
        return None


def get_instant_subtitles(
    video_id: str,
    preferred_lang: str = "en"
) -> Optional[Tuple[List[Dict[str, Any]], str, str, float]]:
    """
    Ultra-fast Subtitle Extraction via YouTubeTranscriptApi in ~1 second.
    Prioritizes manual subtitles, then auto-captions.
    Returns:
        (cleaned_segments, language_code, "youtube_subtitles", estimated_duration) or None
    """
    if not YouTubeTranscriptApi:
        return None

    try:
        api = YouTubeTranscriptApi()
        transcript_list = api.list(video_id)

        manual_track = None
        auto_track = None
        pref_lower = preferred_lang.lower()

        # 1. Search manual tracks first (preferred language, then any)
        for t in transcript_list:
            if not t.is_generated:
                if t.language_code.lower() == pref_lower:
                    manual_track = t
                    break
                elif t.language_code.lower().startswith(f"{pref_lower}-") and not manual_track:
                    manual_track = t

        if not manual_track:
            for t in transcript_list:
                if not t.is_generated:
                    manual_track = t
                    break

        # 2. Search auto-generated tracks
        for t in transcript_list:
            if t.is_generated:
                if t.language_code.lower() == pref_lower:
                    auto_track = t
                    break
                elif t.language_code.lower().startswith(f"{pref_lower}-") and not auto_track:
                    auto_track = t

        if not auto_track:
            for t in transcript_list:
                if t.is_generated:
                    auto_track = t
                    break

        chosen_track = manual_track or auto_track
        if chosen_track:
            snippets = chosen_track.fetch()
            raw_segments = []
            max_end = 0.0
            for s in snippets:
                s_start = round(float(s.start), 2)
                s_dur = float(s.duration or 1.0)
                s_end = round(s_start + s_dur, 2)
                if s_end > max_end:
                    max_end = s_end
                raw_segments.append({
                    "start": s_start,
                    "end": s_end,
                    "text": s.text,
                })

            cleaned = deduplicate_segments(raw_segments)
            if cleaned:
                logger.info(f"Instant transcript found via YouTubeTranscriptApi ({chosen_track.language_code}) with {len(cleaned)} segments")
                return cleaned, chosen_track.language_code, "youtube_subtitles", max_end
    except Exception as e:
        logger.info(f"Instant subtitle fast-path missed/failed ({video_id}): {e}. Falling back to yt-dlp.")

    return None


def get_ydl_base_opts() -> Dict[str, Any]:
    """Base yt-dlp configuration with security, timeouts, client emulation, and JS runtime support."""
    opts: Dict[str, Any] = {
        "quiet": True,
        "no_warnings": True,
        "extract_flat": False,
        "socket_timeout": 20,
        "nocheckcertificate": True,
        "noplaylist": True,
        "http_headers": {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
            "Accept-Language": "en-US,en;q=0.9",
        },
        # Use android, ios, mweb clients to bypass YouTube 403 Forbidden streaming blocks on datacenter IPs
        "extractor_args": {
            "youtube": {
                "player_client": ["android", "ios", "mweb", "web"],
            }
        },
    }

    # Enable Node.js JS runtime if node is in PATH
    if shutil.which("node"):
        opts["js_runtimes"] = {"node": {}}

    # Optional cookiefile support for cloud deployments (e.g. Streamlit Cloud)
    cookies_path = os.getenv("YOUTUBE_COOKIES_PATH", "cookies.txt")
    if os.path.exists(cookies_path):
        opts["cookiefile"] = cookies_path

    return opts


def get_video_info(url_or_id: str) -> Dict[str, Any]:
    """
    Extract video metadata and available subtitle tracks without downloading media.
    Translates yt-dlp exceptions into user-friendly typed errors.
    """
    is_valid, video_id, canonical_url = validate_youtube_url(url_or_id)
    if not is_valid or not canonical_url:
        raise VideoUnavailableError("Invalid YouTube URL or video ID. Please check the URL and try again.")

    ydl_opts = get_ydl_base_opts()
    ydl_opts["skip_download"] = True

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(canonical_url, download=False)
            if not info:
                raise VideoUnavailableError("Could not retrieve video information. The video may be unavailable.")
            return info
    except yt_dlp.utils.DownloadError as e:
        err_msg = str(e).lower()
        logger.warning(f"yt-dlp DownloadError for {canonical_url}: {e}")
        if "private video" in err_msg or "this video is private" in err_msg:
            raise PrivateVideoError("This video is private and cannot be accessed.")
        elif "sign in to confirm your age" in err_msg or "age-restricted" in err_msg:
            raise AgeRestrictedError("This video is age-restricted and requires authentication.")
        elif "video unavailable" in err_msg or "this video has been removed" in err_msg:
            raise VideoUnavailableError("This video is unavailable or has been removed.")
        elif "not available in your country" in err_msg:
            raise VideoUnavailableError("This video is not available in your region.")
        else:
            raise VideoUnavailableError(f"Unable to access YouTube video: {clean_caption_text(str(e))}")
    except Exception as e:
        logger.error(f"Unexpected error extracting video info: {e}", exc_info=True)
        raise YouTubeError("An error occurred while fetching video details.")


def _get_candidate_tracks(caption_dict: Dict[str, List[Dict[str, Any]]], preferred_lang: str = "en") -> List[Tuple[str, Dict[str, Any]]]:
    """
    Rank and return all available subtitle tracks in priority order:
    1. Preferred language native/master track (no 'tlang=' in URL)
    2. Preferred language auto-translated track
    3. Master original language tracks (no 'tlang=' in URL or lang code ending in '-orig')
    4. Any other available language tracks
    Prefers json3 format, followed by vtt, srv3, and srt.
    """
    if not caption_dict:
        return []

    format_preference = ["json3", "vtt", "srv3", "srv1", "ttml", "srt"]

    def fmt_rank(f: Dict[str, Any]) -> int:
        ext = f.get("ext", "").lower()
        if ext in format_preference:
            return format_preference.index(ext)
        return len(format_preference) + 1

    preferred_lower = preferred_lang.lower()
    pref_native: List[Tuple[str, Dict[str, Any]]] = []
    pref_translated: List[Tuple[str, Dict[str, Any]]] = []
    master_native: List[Tuple[str, Dict[str, Any]]] = []
    others: List[Tuple[str, Dict[str, Any]]] = []

    for lang, formats in caption_dict.items():
        if not formats:
            continue
        sorted_fmts = sorted(formats, key=fmt_rank)
        for fmt in sorted_fmts:
            url = fmt.get("url")
            if not url:
                continue
            is_translated = "tlang=" in url
            is_pref = (lang.lower() == preferred_lower or 
                       lang.lower().startswith(f"{preferred_lower}-") or 
                       lang.lower().startswith(f"{preferred_lower}."))
            
            if is_pref and not is_translated:
                pref_native.append((lang, fmt))
            elif is_pref and is_translated:
                pref_translated.append((lang, fmt))
            elif not is_translated or lang.endswith("-orig"):
                master_native.append((lang, fmt))
            else:
                others.append((lang, fmt))
            break  # pick best format per language

    return pref_native + pref_translated + master_native + others


def _download_and_parse_subtitle(track_info: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Download subtitle data directly via HTTP and parse into segments with auto-fallback."""
    url = track_info.get("url")
    ext = track_info.get("ext", "").lower()
    if not url:
        return []

    def fetch_url(target_url: str) -> Optional[str]:
        req = urllib.request.Request(
            target_url,
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
                "Referer": "https://www.youtube.com/",
            }
        )
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                return resp.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as e:
            logger.debug(f"HTTPError {e.code} fetching subtitles from {target_url[:60]}")
            return None
        except Exception as e:
            logger.debug(f"Error fetching subtitles: {e}")
            return None

    content = fetch_url(url)

    # If fetching failed and URL contained auto-translation (tlang=), strip tlang parameter to get the original spoken track!
    if not content and "tlang=" in url:
        try:
            parsed = urllib.parse.urlparse(url)
            qs = urllib.parse.parse_qs(parsed.query)
            qs.pop("tlang", None)
            clean_query = urllib.parse.urlencode(qs, doseq=True)
            fallback_url = urllib.parse.urlunparse(parsed._replace(query=clean_query))
            logger.info("Retrying subtitle fetch by removing tlang parameter...")
            content = fetch_url(fallback_url)
        except Exception as err:
            logger.debug(f"Failed to strip tlang parameter: {err}")

    if not content:
        return []

    if ext == "json3" or "timedtext" in url:
        try:
            data = json.loads(content)
            if isinstance(data, dict) and "events" in data:
                return parse_json3_content(data)
        except Exception:
            pass

    return parse_vtt_content(content)


def extract_youtube_subtitles(info: Dict[str, Any], preferred_lang: str = "en") -> Optional[Tuple[List[Dict[str, Any]], str, str]]:
    """
    Subtitle-First Strategy:
    1. Check for manually created subtitles ('subtitles')
    2. Check for automatically generated subtitles ('automatic_captions')
    3. Iterate through candidate tracks until a usable track is downloaded.
    
    Returns:
        (segments, language_code, "youtube_subtitles") or None
    """
    # 1. Check manually created subtitles
    manual_subs = info.get("subtitles") or {}
    for lang, track_info in _get_candidate_tracks(manual_subs, preferred_lang):
        segments = _download_and_parse_subtitle(track_info)
        if segments:
            logger.info(f"Successfully extracted {len(segments)} segments from manual subtitles ({lang})")
            return segments, lang, "youtube_subtitles"

    # 2. Check automatically generated subtitles
    auto_subs = info.get("automatic_captions") or {}
    for lang, track_info in _get_candidate_tracks(auto_subs, preferred_lang):
        segments = _download_and_parse_subtitle(track_info)
        if segments:
            logger.info(f"Successfully extracted {len(segments)} segments from auto captions ({lang})")
            return segments, lang, "youtube_subtitles"

    logger.info("No usable YouTube subtitles found. Falling back to audio download.")
    return None


def download_audio(
    canonical_url: str,
    output_dir: str,
    progress_callback: Optional[Callable[[int, str], None]] = None
) -> str:
    """
    Download only the audio stream of a YouTube video as an optimized MP3 file.
    Uses android/ios player clients and format fallback to avoid HTTP 403 Forbidden.
    Returns:
        Absolute filepath to the downloaded audio file.
    """
    os.makedirs(output_dir, exist_ok=True)
    out_template = os.path.join(output_dir, "%(id)s_%(epoch)s.%(ext)s")

    def ydl_progress_hook(d: Dict[str, Any]) -> None:
        if d.get("status") == "downloading" and progress_callback:
            total = d.get("total_bytes") or d.get("total_bytes_estimate") or 0
            downloaded = d.get("downloaded_bytes") or 0
            if total > 0:
                pct = int((downloaded / total) * 100)
                progress_callback(pct, f"Downloading audio: {pct}% complete...")

    ydl_opts = get_ydl_base_opts()
    ydl_opts.update({
        "format": "ba[ext=m4a]/ba/b/18/best",
        "outtmpl": out_template,
        "progress_hooks": [ydl_progress_hook],
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "mp3",
            "preferredquality": "128",
        }],
    })

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            res = ydl.extract_info(canonical_url, download=True)
            video_id = res.get("id")

            # Find the downloaded file
            for fname in os.listdir(output_dir):
                if fname.startswith(f"{video_id}_") and fname.endswith(".mp3"):
                    return os.path.abspath(os.path.join(output_dir, fname))

            # Fallback check any matching files
            for fname in os.listdir(output_dir):
                if fname.startswith(f"{video_id}_"):
                    return os.path.abspath(os.path.join(output_dir, fname))

            raise YouTubeError("Audio was downloaded but the output file could not be located.")
    except yt_dlp.utils.DownloadError as e:
        logger.error(f"Failed to download audio for {canonical_url}: {e}")
        err_msg = str(e)
        if "403" in err_msg:
            raise YouTubeError(
                "YouTube blocked audio stream download (HTTP 403 Forbidden). "
                "Cloud datacenter IPs (like Streamlit Cloud) may require a cookies.txt file to download raw audio streams."
            )
        raise YouTubeError(f"Audio download failed: {clean_caption_text(err_msg)}")
    except Exception as e:
        logger.error(f"Unexpected audio download error: {e}", exc_info=True)
        raise YouTubeError(f"Audio extraction failed: {str(e)}")
