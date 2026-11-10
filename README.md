# Voice to Notes

A local web app that turns a recorded lecture into a textbook-style chapter.

1. **Transcribe** — Whisper (`faster-whisper`) runs on your NVIDIA GPU through CUDA.
2. **Compose** — a local language model served by [Ollama](https://ollama.com) rewrites the transcript
   as structured textbook prose: sections, definitions, examples, LaTeX maths, plus an overview,
   learning objectives, key takeaways, a glossary and review questions.
3. **Export** — the chapter is saved as Markdown and rendered to a typeset PDF.

Everything runs on your own machine; no audio or text leaves it.

## Sessions

Every run is a **session** with its own folder:

```
output/<session>/
    <session>.md      the textbook chapter
    <session>.pdf     the typeset PDF
    transcript.txt    the raw transcript, with timestamps
```

A session can contain **several recordings** (for example a lecture recorded in parts). Drop them all,
then drag the rows in the list into the right order (or use the ↑ ↓ buttons). They are transcribed in
that order and joined into one chapter.

## Languages

- **Auto-detect (per recording)** — each recording's language is detected separately, so a session can
  mix recordings in different languages.
- **Multilingual** — for recordings where the speaker switches language *within* the recording; the
  language is detected for every segment.
- Or pick one fixed spoken language.

The notes are written in one language: by default the language spoken most in the session, or any
language you choose under **Notes language**. Passages in other languages are translated.

## Markdown → PDF

The app also converts any other Markdown file to PDF: drop it on the **Markdown → PDF** card (it gets
its own folder in `output/`), or use **Convert to PDF** next to a session in the list.

## Requirements

- Windows with an NVIDIA GPU and a current driver (falls back to CPU if CUDA is unavailable)
- [uv](https://docs.astral.sh/uv/)
- [Ollama](https://ollama.com) with at least one model pulled, e.g. `ollama pull qwen3:14b`

## Setup

```powershell
powershell -ExecutionPolicy Bypass -File setup.ps1
```

This creates the Python 3.12 environment (including the CUDA libraries), downloads a headless Chromium
for PDF rendering, and fetches KaTeX for maths.

## Run

Double-click `run.bat`, or:

```powershell
uv run python -m app
```

The app opens at <http://127.0.0.1:8765>. The first transcription downloads the chosen Whisper model.

## Notes

- Supported input: mp3, wav, m4a, aac, flac, ogg, opus, wma, webm, mp4, mkv, mov — or a `.txt`
  transcript, which skips transcription (useful for re-composing a session's `transcript.txt` with another model).
- Whisper is unloaded before composing so the language model gets the VRAM.
- Choose **None — cleaned transcript only** as the writing model to skip the language model.
- Environment variables: `PORT` (default 8765), `OLLAMA_URL` (default `http://127.0.0.1:11434`),
  `NO_BROWSER=1` to not open a browser tab on start.
