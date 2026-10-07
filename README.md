# YouTube Full Transcript Extractor (Streamlit Edition)

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Web%20App-Streamlit-FF4B4B.svg)](https://streamlit.io/)
[![yt-dlp](https://img.shields.io/badge/Extractor-yt--dlp-red.svg)](https://github.com/yt-dlp/yt-dlp)
[![faster-whisper](https://img.shields.io/badge/AI%20STT-faster--whisper-purple.svg)](https://github.com/SYSTRAN/faster-whisper)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A high-performance web application built with **Streamlit** to extract the **complete, untruncated transcript** of any YouTube video.

Powered by an intelligent **Subtitle-First Strategy** and a local **`faster-whisper` AI fallback**. Completely free with **zero paid API keys required**.

---

## 1. Features

- **Streamlit Web Interface:** Clean, modern, responsive dark-themed dashboard.
- **Subtitle-First Intelligence:** Instant extraction of creator & auto-generated captions in ~1-2 seconds.
- **Local AI Fallback:** Automatically downloads audio-only stream and transcribes via `faster-whisper` when captions are unavailable.
- **Zero Truncation:** Delivers 100% complete spoken video transcripts with timestamps.
- **Live Progress Reporting:** Uses Streamlit status and progress indicators for real-time visibility.
- **In-Memory Caching:** Automatically caches transcripts for sub-millisecond repeat queries.
- **Interactive Multi-Tab Viewer:**
  - **⏱️ Timestamped Segments:** Clean cards with `[HH:MM:SS - HH:MM:SS]` badges.
  - **📄 Continuous Full Text:** Read entire transcript continuously with one-click copy.
  - **📦 Raw JSON Inspector:** Interactive JSON data tree.
- **In-Transcript Search:** Instant real-time filtering and keyword highlighting.
- **One-Click Exports:**
  - **Download JSON** (Complete metadata + transcript array matching specification)
  - **Download TXT**
  - **Download SRT**
- **Hardware Acceleration:** Auto-detects NVIDIA CUDA GPUs (`float16`), with CPU fallback (`int8`).
- **Temporary File Cleanup:** Automatically deletes any temporary audio downloads immediately after transcription.

---

## 2. Quick Start (Windows)

### Step 1: Open PowerShell in Project Root
```powershell
cd "d:\YouTube Transcript Extractor"
```

### Step 2: Activate Virtual Environment
```powershell
.\venv\Scripts\Activate.ps1
```

### Step 3: Install Dependencies
```powershell
pip install -r requirements.txt
```

### Step 4: Run the Streamlit Application
```powershell
streamlit run app.py
```

The application will open automatically in your browser at:
👉 **[http://localhost:8501](http://localhost:8501)**

---

## 3. Project Structure

```text
YouTube Transcript Extractor/
│
├── app.py                      # Main Streamlit Web Application
│
├── backend/
│   ├── __init__.py             # Python package marker
│   ├── transcript_service.py   # Multi-tier orchestrator (Fast-Path, Cache, Fallbacks)
│   ├── youtube_service.py      # Subtitle extraction & yt-dlp audio download
│   ├── whisper_service.py      # faster-whisper singleton manager & caching
│   ├── models.py               # Pydantic data schemas
│   └── utils.py                # Formatting, deduplication, URL validation
│
├── .streamlit/
│   └── config.toml             # Custom dark theme configuration
│
├── tests/
│   └── test_app.py             # Automated test suite (pytest)
│
├── temp/                       # Temporary audio directory (auto-cleaned)
├── downloads/                  # Storage for exports
├── requirements.txt            # Project dependencies
├── .env.example                # Environment sample
├── .gitignore                  # Git ignore rules
└── README.md                   # Documentation
```

---

## 4. Whisper Model Selection

When a video has no subtitles on YouTube, the system uses `faster-whisper`:

| Model | Disk Size | Speed | Recommendation |
| :--- | :--- | :--- | :--- |
| `tiny` | ~75 MB | ~32x | Fastest, minimal RAM (~1 GB) |
| `base` | ~145 MB | ~16x | Fast, good for clear speech (~1.5 GB) |
| `small` | ~480 MB | ~6x | **Default (Recommended)**: Great balance (~2 GB) |
| `medium` | ~1.5 GB | ~2x | High accuracy for accents/technical words (~5 GB) |
| `large-v3` | ~3.1 GB | ~1x | Maximum quality across multilingual speech (~10 GB) |

---

## 5. Downloaded JSON Format

```json
{
  "success": true,
  "video_id": "jNQXAC9IVRw",
  "video_url": "https://www.youtube.com/watch?v=jNQXAC9IVRw",
  "title": "Me at the zoo",
  "channel": "jawed",
  "language": "en",
  "duration": 19.0,
  "duration_formatted": "00:00:19",
  "thumbnail": "https://i.ytimg.com/vi/jNQXAC9IVRw/hqdefault.jpg",
  "method": "youtube_subtitles",
  "transcription_method": "youtube_subtitles",
  "model_used": null,
  "word_count": 52,
  "segment_count": 6,
  "transcript": [
    {
      "start": 1.2,
      "end": 3.36,
      "text": "All right, so here we are, in front of the elephants"
    }
  ]
}
```

---

## 6. Running Tests

```powershell
python -m pytest tests/test_app.py -v
```
