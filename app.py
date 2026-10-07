import os
import sys
import json
import asyncio
import streamlit as st

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backend.utils import (
    validate_youtube_url,
    format_timestamp,
    check_ffmpeg_installed,
)
from backend.whisper_service import (
    whisper_manager,
    SUPPORTED_MODELS,
    DEFAULT_MODEL,
)
from backend.transcript_service import (
    stream_transcript_progress,
    _TRANSCRIPT_CACHE,
)

# 1. Page Configuration
st.set_page_config(
    page_title="YouTube Full Transcript Extractor",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 2. Custom CSS for Modern Dark Aesthetics
st.markdown("""
<style>
    /* Global Styles */
    .stApp {
        background-color: #07090e;
        color: #f8fafc;
    }
    
    /* Headers & Typography */
    h1, h2, h3 {
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        letter-spacing: -0.02em;
    }
    
    .gradient-header {
        background: linear-gradient(135deg, #ffffff 10%, #38bdf8 50%, #818cf8 90%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.2rem;
        font-weight: 800;
        margin-bottom: 4px;
    }
    
    .header-subtext {
        color: #94a3b8;
        font-size: 1rem;
        margin-bottom: 24px;
    }
    
    /* Metric & Badge Pills */
    .badge-pill {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 700;
        letter-spacing: 0.04em;
    }
    
    .badge-subtitles {
        background: rgba(16, 185, 129, 0.15);
        color: #34d399;
        border: 1px solid rgba(16, 185, 129, 0.4);
    }
    
    .badge-whisper {
        background: rgba(129, 140, 248, 0.2);
        color: #a5b4fc;
        border: 1px solid rgba(129, 140, 248, 0.4);
    }
    
    /* Segment Card */
    .segment-card {
        background: rgba(16, 23, 38, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 10px;
        padding: 12px 16px;
        margin-bottom: 10px;
        display: flex;
        align-items: flex-start;
        gap: 14px;
        transition: all 0.2s ease;
    }
    
    .segment-card:hover {
        background: rgba(23, 33, 54, 0.85);
        border-color: rgba(56, 189, 248, 0.3);
    }
    
    .segment-time {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.8rem;
        font-weight: 600;
        color: #38bdf8;
        background: rgba(56, 189, 248, 0.1);
        border: 1px solid rgba(56, 189, 248, 0.25);
        padding: 3px 8px;
        border-radius: 6px;
        white-space: nowrap;
    }
    
    .segment-text {
        font-size: 0.95rem;
        line-height: 1.6;
        color: #e2e8f0;
        flex: 1;
    }
    
    /* Highlight for search match */
    mark.highlight {
        background-color: rgba(245, 158, 11, 0.4);
        color: #fef08a;
        padding: 1px 4px;
        border-radius: 3px;
        font-weight: 600;
    }
    
    /* Card Container */
    .info-card {
        background: rgba(16, 23, 38, 0.75);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 18px;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)


# 3. Sidebar: Configuration & System Status
with st.sidebar:
    st.markdown("### ⚙️ System & Settings")
    
    # Hardware Status Pill
    if whisper_manager.cuda_available:
        st.success(f"🚀 CUDA GPU Active ({whisper_manager.compute_type})")
    else:
        st.info(f"⚡ CPU Active ({whisper_manager.compute_type})")
    
    ffmpeg_ok = check_ffmpeg_installed()
    if ffmpeg_ok:
        st.caption("✅ FFmpeg: Installed & Ready")
    else:
        st.warning("⚠️ FFmpeg: Not found in PATH (required for audio fallback)")
        
    st.markdown("---")
    
    # Whisper Model Selection
    st.markdown("#### 🧠 Whisper AI Fallback Model")
    st.caption("Used automatically when a video has zero subtitles.")
    
    model_choice = st.selectbox(
        "Model Size",
        options=SUPPORTED_MODELS,
        index=SUPPORTED_MODELS.index(DEFAULT_MODEL),
        format_func=lambda m: {
            "tiny": "tiny (~75MB, ~32x speed)",
            "base": "base (~145MB, ~16x speed)",
            "small": "small (~480MB, Default - Recommended)",
            "medium": "medium (~1.5GB, High Accuracy)",
            "large-v3": "large-v3 (~3.1GB, Max Quality)",
        }.get(m, m)
    )
    
    preferred_lang = st.text_input(
        "Preferred Subtitle Language",
        value="en",
        help="Language code to prioritize for subtitles (e.g. 'en', 'mr', 'hi', 'es')."
    ).strip() or "en"
    
    st.markdown("---")
    
    # Cache Control
    if st.button("🧹 Clear In-Memory Cache", width="stretch"):
        _TRANSCRIPT_CACHE.clear()
        st.session_state.transcript_data = None
        st.session_state.youtube_url_input = ""
        st.toast("In-memory cache cleared!", icon="✅")
        
    st.markdown("---")
    with st.expander("ℹ️ About This App"):
        st.markdown("""
        **YouTube Full Transcript Extractor**
        - **Subtitle-First:** Extracts creator & auto-generated captions in ~1-2 seconds.
        - **Local AI:** Transcribes missing audio with `faster-whisper`.
        - **100% Free:** Zero paid API keys required.
        - **No Truncation:** Returns the entire spoken video transcript.
        """)


# 4. Main Page Header
st.markdown('<div class="gradient-header">YouTube Full Transcript Extractor</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="header-subtext">Subtitle-first intelligence with <code>faster-whisper</code> local AI fallback. Zero truncation, zero summarization.</div>',
    unsafe_allow_html=True
)

# Initialize Session State
if "transcript_data" not in st.session_state:
    st.session_state.transcript_data = None
if "youtube_url_input" not in st.session_state:
    st.session_state.youtube_url_input = ""


# 5. URL Input & Controls
with st.form("transcript_form", clear_on_submit=False, border=False):
    col_input, col_btn = st.columns([5, 1])

    with col_input:
        url_input = st.text_input(
            "YouTube Video URL or Video ID",
            key="youtube_url_input",
            placeholder="https://www.youtube.com/watch?v=... or https://youtu.be/...",
            label_visibility="collapsed"
        ).strip()

    with col_btn:
        extract_clicked = st.form_submit_button("⚡ Get Transcript", type="primary", width="stretch")


# 6. Extraction Execution
if extract_clicked:
    if not url_input:
        st.session_state.transcript_data = None
        st.error("Please enter a YouTube video URL or ID.")
    else:
        is_valid, video_id, canonical_url = validate_youtube_url(url_input)
        if not is_valid:
            st.session_state.transcript_data = None
            st.error("Invalid YouTube URL. Please provide a standard watch link, shorts, or 11-character video ID.")
        else:
            # Clear previous transcript immediately so stale results never persist
            st.session_state.transcript_data = None
            
            # Real-time extraction with Streamlit status
            with st.status("Extracting Full Transcript...", expanded=True) as status_box:
                progress_bar = st.progress(5)
                
                async def execute_stream():
                    result = None
                    async for event in stream_transcript_progress(
                        url=canonical_url,
                        model_name=model_choice,
                        preferred_lang=preferred_lang,
                    ):
                        stage = event.get("stage")
                        msg = event.get("message", "Processing...")
                        pct = event.get("percent", 10)
                        
                        status_box.write(f"• **{msg}**")
                        progress_bar.progress(min(100, max(5, pct)))
                        
                        if stage == "complete" and "data" in event:
                            result = event["data"]
                        elif stage == "error":
                            raise RuntimeError(event.get("message", "Extraction failed"))
                    return result
                
                try:
                    data = asyncio.run(execute_stream())
                    if data:
                        st.session_state.transcript_data = data
                        status_box.update(label="✅ Transcript Extracted Successfully!", state="complete", expanded=False)
                        st.toast("Transcript ready!", icon="🎉")
                except Exception as err:
                    status_box.update(label="❌ Extraction Error", state="error", expanded=True)
                    st.error(f"Error: {str(err)}")


# 7. Render Results
data = st.session_state.transcript_data

if data:
    st.markdown("---")
    
    # Video Meta Card
    meta_col1, meta_col2 = st.columns([1, 3])
    with meta_col1:
        thumbnail_url = data.get("thumbnail") or f"https://i.ytimg.com/vi/{data.get('video_id')}/hqdefault.jpg"
        st.image(thumbnail_url, width="stretch")
        
    with meta_col2:
        st.subheader(data.get("title", "Untitled Video"))
        st.caption(f"Channel: {data.get('channel') or 'YouTube'} • URL: {data.get('video_url')}")
        
        # Badges & Metrics
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Duration", data.get("duration_formatted", "00:00:00"))
        
        method_name = data.get("transcription_method", "youtube_subtitles")
        is_subtitles = method_name == "youtube_subtitles"
        method_display = "YouTube Subtitles" if is_subtitles else f"Whisper ({data.get('model_used') or model_choice})"
        m2.metric("Method", method_display)
        
        m3.metric("Language", (data.get("language") or "en").upper())
        m4.metric("Word Count", f"{data.get('word_count', 0):,}")
        m5.metric("Segments", f"{data.get('segment_count', 0):,}")

    st.markdown("---")
    
    # Format Content for Downloads
    transcript_segments = data.get("transcript", [])
    
    # 1. JSON Content (Exact Section 7 Format)
    json_export = json.dumps(data, indent=2, ensure_ascii=False)
    
    # 2. Formatted TXT Content
    txt_lines = [
        f"Title: {data.get('title')}",
        f"URL: {data.get('video_url')}",
        f"Duration: {data.get('duration_formatted')}",
        f"Method: {data.get('transcription_method')}",
        f"Language: {data.get('language')}",
        f"Total Words: {data.get('word_count')}",
        "-" * 50,
        ""
    ]
    for seg in transcript_segments:
        s_time = format_timestamp(seg.get("start", 0))
        e_time = format_timestamp(seg.get("end", 0))
        txt_lines.append(f"[{s_time} - {e_time}]\n{seg.get('text')}\n")
    txt_export = "\n".join(txt_lines)
    
    # 3. Standard SRT Content
    srt_blocks = []
    for idx, seg in enumerate(transcript_segments, 1):
        s_time = format_timestamp(seg.get("start", 0), include_ms=True).replace(".", ",")
        e_time = format_timestamp(seg.get("end", 0), include_ms=True).replace(".", ",")
        srt_blocks.append(f"{idx}\n{s_time} --> {e_time}\n{seg.get('text')}\n")
    srt_export = "\n".join(srt_blocks)
    
    # 4. Continuous Prose Text
    full_prose_text = " ".join(seg.get("text", "") for seg in transcript_segments)

    # Action Toolbar: Prominent Download Buttons
    btn_col1, btn_col2, btn_col3, btn_col4 = st.columns([2, 1, 1, 1])
    
    video_slug = "".join(c if c.isalnum() else "_" for c in data.get("title", data.get("video_id", "transcript")))[:50]
    
    with btn_col1:
        st.download_button(
            label="📥 Download JSON (Full Metadata)",
            data=json_export,
            file_name=f"{video_slug}_transcript.json",
            mime="application/json",
            type="primary",
            width="stretch"
        )
    with btn_col2:
        st.download_button(
            label="📄 Download TXT",
            data=txt_export,
            file_name=f"{video_slug}_transcript.txt",
            mime="text/plain",
            width="stretch"
        )
    with btn_col3:
        st.download_button(
            label="⏱️ Download SRT",
            data=srt_export,
            file_name=f"{video_slug}_subtitles.srt",
            mime="application/x-subrip",
            width="stretch"
        )
    with btn_col4:
        if st.button("📋 Copy Text", width="stretch"):
            st.session_state["copied"] = True
            st.toast("Full transcript text ready to copy below!", icon="📋")

    # In-Transcript Search Filter
    st.markdown("#### 🔍 Filter & Search Transcript")
    search_query = st.text_input(
        "Search spoken words...",
        placeholder="Type words to search and highlight in transcript...",
        label_visibility="collapsed"
    ).strip().lower()

    # Content Display Tabs
    tab_segments, tab_fulltext, tab_json = st.tabs(["⏱️ Timestamped Segments", "📄 Continuous Full Text", "📦 Raw JSON Inspector"])
    
    with tab_segments:
        filtered_segments = []
        for seg in transcript_segments:
            text = seg.get("text", "")
            if not search_query or search_query in text.lower():
                filtered_segments.append(seg)
                
        if search_query:
            st.caption(f"Showing **{len(filtered_segments)}** matching segment(s) for '{search_query}'")
            
        if not filtered_segments:
            st.info("No segments match your search query.")
        else:
            # Render segments in a clean scrollable container
            segments_html = []
            for seg in filtered_segments:
                s_time = format_timestamp(seg.get("start", 0))
                e_time = format_timestamp(seg.get("end", 0))
                seg_text = seg.get("text", "")
                
                if search_query:
                    # Highlight occurrences
                    import re
                    pattern = re.compile(re.escape(search_query), re.IGNORECASE)
                    seg_text = pattern.sub(lambda m: f'<mark class="highlight">{m.group(0)}</mark>', seg_text)
                    
                segments_html.append(f"""
                <div class="segment-card">
                    <span class="segment-time">{s_time} - {e_time}</span>
                    <span class="segment-text">{seg_text}</span>
                </div>
                """)
            st.markdown(f'<div style="max-height: 520px; overflow-y: auto; padding-right: 8px;">{"".join(segments_html)}</div>', unsafe_allow_html=True)
            
    with tab_fulltext:
        st.text_area(
            "Continuous Full Text",
            value=full_prose_text,
            height=420,
            label_visibility="collapsed"
        )
        
    with tab_json:
        st.json(data)
