"""GPU speech-to-text with faster-whisper (CTranslate2 on CUDA)."""
from __future__ import annotations

import gc
import os
import re
from pathlib import Path
from typing import Callable

WHISPER_MODELS = ["large-v3-turbo", "large-v3", "medium", "small"]

_model = None
_model_key: tuple[str, str] | None = None
_dll_ready = False


def _add_cuda_dll_dirs() -> None:
    """Make the pip-installed cuBLAS / cuDNN DLLs visible to CTranslate2 on Windows."""
    global _dll_ready
    if _dll_ready or os.name != "nt":
        return
    _dll_ready = True
    try:
        import nvidia  # namespace package from the nvidia-*-cu12 wheels
    except ImportError:
        return
    for base in list(nvidia.__path__):
        for bin_dir in Path(base).glob("*/bin"):
            os.environ["PATH"] = f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}"
            try:
                os.add_dll_directory(str(bin_dir))
            except OSError:
                pass


def cuda_available() -> bool:
    _add_cuda_dll_dirs()
    try:
        import ctranslate2

        return ctranslate2.get_cuda_device_count() > 0
    except Exception:
        return False


def unload() -> None:
    """Free the Whisper model so the VRAM is available to the language model."""
    global _model, _model_key
    _model = None
    _model_key = None
    gc.collect()


def _load(name: str, log: Callable[[str], None]):
    global _model, _model_key
    if _model is not None and _model_key and _model_key[0] == name:
        return _model, _model_key[1]
    unload()
    _add_cuda_dll_dirs()
    from faster_whisper import WhisperModel

    device = "cpu"
    model = None
    if cuda_available():
        try:
            model = WhisperModel(name, device="cuda", compute_type="float16")
            device = "cuda"
        except Exception as exc:  # fall back rather than fail the whole job
            log(f"GPU load failed ({exc}); falling back to CPU.")
    if model is None:
        model = WhisperModel(name, device="cpu", compute_type="int8")
    _model, _model_key = model, (name, device)
    return model, device


def _to_paragraphs(segments: list[dict]) -> list[dict]:
    """Group Whisper segments into readable paragraphs."""
    paragraphs: list[dict] = []
    cur: list[dict] = []
    words = 0
    for seg in segments:
        if cur:
            gap = seg["start"] - cur[-1]["end"]
            ended = cur[-1]["text"].rstrip().endswith((".", "?", "!", "…"))
            if (gap >= 2.0 and words >= 40) or (words >= 130 and ended) or words >= 220:
                paragraphs.append({"start": cur[0]["start"], "text": " ".join(s["text"] for s in cur)})
                cur, words = [], 0
        cur.append(seg)
        words += len(seg["text"].split())
    if cur:
        paragraphs.append({"start": cur[0]["start"], "text": " ".join(s["text"] for s in cur)})
    return paragraphs


def _stamp(seconds: float) -> str:
    s = int(seconds)
    return f"{s // 3600:02d}:{s % 3600 // 60:02d}:{s % 60:02d}"


def transcribe(
    path: Path,
    model_name: str,
    language: str | None,
    multilingual: bool,
    on_progress: Callable[[float, str], None],
    should_cancel: Callable[[], bool],
    log: Callable[[str], None],
) -> dict:
    log(f"Loading Whisper model '{model_name}' (downloaded on first use)…")
    model, device = _load(model_name, log)
    log(f"Whisper running on {'GPU (CUDA, float16)' if device == 'cuda' else 'CPU (int8)'}.")

    from faster_whisper import BatchedInferencePipeline

    pipeline = BatchedInferencePipeline(model=model)
    seg_iter, info = pipeline.transcribe(
        str(path),
        language=None if multilingual else (language or None),
        multilingual=multilingual,  # detect the language of every segment (mixed-language speech)
        batch_size=16 if device == "cuda" else 4,
        vad_filter=True,
    )
    duration = max(info.duration or 0.0, 0.001)
    lang_note = f"multilingual, mainly {info.language}" if multilingual else f"language: {info.language}"
    log(f"Audio length {_stamp(duration)}, {lang_note}.")

    segments: list[dict] = []
    for seg in seg_iter:
        if should_cancel():
            raise InterruptedError
        text = seg.text.strip()
        if text:
            segments.append({"start": seg.start, "end": seg.end, "text": text})
        on_progress(min(seg.end / duration, 1.0), f"{_stamp(seg.end)} / {_stamp(duration)}")

    paragraphs = _to_paragraphs(segments)
    return {
        "language": info.language,
        "duration": duration,
        "device": device,
        "paragraphs": [p["text"] for p in paragraphs],
        "transcript_text": "\n\n".join(f"[{_stamp(p['start'])}] {p['text']}" for p in paragraphs),
    }


def paragraphs_from_text(text: str) -> list[str]:
    """Read a plain-text transcript (e.g. one saved by a previous run)."""
    out = []
    for block in re.split(r"\n\s*\n", text):
        block = block.strip()
        if re.fullmatch(r"=== .* ===", block):  # "=== Part 2: file.mp3 ===" separator
            continue
        block = re.sub(r"^\[\d{1,2}:\d{2}(:\d{2})?\]\s*", "", block)
        block = " ".join(block.split())
        if block:
            out.append(block)
    return out
