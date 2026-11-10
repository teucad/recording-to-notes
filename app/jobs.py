"""The lecture pipeline: transcribe -> compose -> write Markdown -> export PDF.

Every session gets its own folder: output/<session>/ with <session>.md, <session>.pdf
and transcript.txt. A session can consist of several recordings, processed in the given order.
"""
from __future__ import annotations

import re
import threading
import time
import traceback
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from . import OUTPUT_DIR
from . import compose as composer
from . import pdf, transcribe

# share of the overall progress bar per stage
WEIGHTS = {"transcribe": (0.0, 0.45), "compose": (0.45, 0.47), "pdf": (0.92, 0.08)}
TEXT_EXTS = {".txt"}
TRANSCRIPT_NAME = "transcript.txt"
MULTILINGUAL = "multi"


@dataclass
class Job:
    sources: list[Path]        # uploaded files, in the order chosen by the user
    source_names: list[str]
    stem: str                  # session folder name and base name of the .md / .pdf
    title: str
    language: str              # spoken language code, "" = auto-detect per recording, "multi" = mixed within a recording
    out_language: str          # language of the notes, "" = same as the lecture
    whisper_model: str
    llm_model: str             # "" = no LLM, cleaned transcript only
    make_pdf: bool
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:10])
    status: str = "queued"     # queued | running | done | error | cancelled
    stage: str = "queued"
    progress: float = 0.0
    message: str = "Waiting to start"
    device: str = ""
    error: str = ""
    log: list[str] = field(default_factory=list)
    outputs: dict[str, str] = field(default_factory=dict)
    started: float = field(default_factory=time.time)
    finished: float = 0.0
    cancel_flag: bool = False

    @property
    def folder(self) -> Path:
        return OUTPUT_DIR / self.stem

    @property
    def display_name(self) -> str:
        extra = len(self.source_names) - 1
        return self.source_names[0] + (f" + {extra} more" if extra else "")

    def add_log(self, line: str) -> None:
        self.log.append(f"{time.strftime('%H:%M:%S')}  {line}")

    def set_stage(self, stage: str, message: str) -> None:
        self.stage, self.message = stage, message
        self.progress = WEIGHTS[stage][0]
        self.add_log(message)

    def stage_progress(self, fraction: float, message: str) -> None:
        start, span = WEIGHTS[self.stage]
        self.progress = start + span * max(0.0, min(fraction, 1.0))
        self.message = message

    def cancelled(self) -> bool:
        return self.cancel_flag

    def to_dict(self) -> dict:
        return {
            "id": self.id, "name": self.display_name, "folder": self.stem, "status": self.status,
            "stage": self.stage, "progress": round(self.progress, 4), "message": self.message,
            "device": self.device, "error": self.error, "log": self.log[-200:], "outputs": self.outputs,
            "elapsed": round((self.finished or time.time()) - self.started),
            "make_pdf": self.make_pdf, "uses_llm": bool(self.llm_model),
        }


JOBS: dict[str, Job] = {}
_lock = threading.Lock()


def active_job() -> Job | None:
    return next((j for j in JOBS.values() if j.status in ("queued", "running")), None)


def slugify(name: str) -> str:
    slug = re.sub(r"[^\w\-]+", "-", name.strip(), flags=re.UNICODE).strip("-_")
    return slug[:80] or "lecture"


def unique_stem(name: str) -> str:
    """A session folder name that does not exist yet."""
    base = slugify(name)
    stem, i = base, 2
    while (OUTPUT_DIR / stem).exists():
        stem = f"{base}-{i}"
        i += 1
    return stem


def start(job: Job) -> None:
    with _lock:
        if active_job():
            raise RuntimeError("Another session is already being processed.")
        job.folder.mkdir(parents=True)
        JOBS[job.id] = job
    threading.Thread(target=_run, args=(job,), daemon=True).start()


def _transcribe_all(job: Job) -> tuple[list[str], str]:
    """Transcribe every recording in order. Returns (paragraphs, main spoken language)."""
    n = len(job.sources)
    audio_total = sum(s.stat().st_size for s in job.sources if s.suffix.lower() not in TEXT_EXTS) or 1
    done = 0.0
    paragraphs: list[str] = []
    transcript_parts: list[str] = []
    multilingual = job.language == MULTILINGUAL
    fixed_language = "" if multilingual else job.language
    spoken: dict[str, float] = {}  # seconds of audio per detected language
    used_audio = False

    for i, (src, name) in enumerate(zip(job.sources, job.source_names), 1):
        if job.cancelled():
            raise InterruptedError
        label = f"Recording {i} of {n}" if n > 1 else "Recording"
        if src.suffix.lower() in TEXT_EXTS:
            job.add_log(f"{label}: reading transcript file {name}")
            part = transcribe.paragraphs_from_text(src.read_text(encoding="utf-8", errors="replace"))
            text = "\n\n".join(part)
        else:
            job.add_log(f"{label}: transcribing {name}")
            share = src.stat().st_size / audio_total

            def on_progress(fraction: float, msg: str, base=done, share=share, label=label) -> None:
                job.stage_progress(base + share * fraction, f"{label} · {msg}")

            result = transcribe.transcribe(
                src, job.whisper_model, fixed_language, multilingual,
                on_progress, job.cancelled, job.add_log,
            )
            done += share
            used_audio = True
            job.device = result["device"]
            spoken[result["language"]] = spoken.get(result["language"], 0.0) + result["duration"]
            part, text = result["paragraphs"], result["transcript_text"]
        paragraphs += part
        transcript_parts.append(f"=== Part {i}: {name} ===\n\n{text}" if n > 1 else text)

    if used_audio:
        (job.folder / TRANSCRIPT_NAME).write_text("\n\n".join(transcript_parts) + "\n", encoding="utf-8")
        job.outputs["transcript"] = TRANSCRIPT_NAME
        job.add_log(f"Transcript saved: output/{job.stem}/{TRANSCRIPT_NAME}")
        transcribe.unload()  # hand the VRAM to the language model
    if len(spoken) > 1:
        job.add_log("Languages detected: " + ", ".join(sorted(spoken, key=spoken.get, reverse=True)))
    # the notes default to the language that was spoken most
    main_language = max(spoken, key=spoken.get) if spoken else fixed_language
    return paragraphs, main_language


def _run(job: Job) -> None:
    job.status = "running"
    try:
        # 1. speech -> text
        n = len(job.sources)
        job.set_stage("transcribe", f"Transcribing {n} recordings" if n > 1 else "Transcribing")
        paragraphs, detected = _transcribe_all(job)
        if not paragraphs:
            raise composer.ComposeError("No speech was found in the uploaded recordings.")

        # 2. text -> textbook chapter
        fallback_title = Path(job.source_names[0]).stem.replace("_", " ").strip() or "Lecture"
        source_label = ", ".join(job.source_names)
        if job.llm_model:
            job.set_stage("compose", "Composing textbook chapter")
            markdown = composer.compose(
                paragraphs, job.llm_model, job.title, fallback_title, source_label,
                job.out_language or detected, job.stage_progress, job.cancelled, job.add_log,
            )
        else:
            job.set_stage("compose", "Formatting transcript (no language model selected)")
            markdown = composer.plain_markdown(paragraphs, job.title or fallback_title, source_label)
        md_path = job.folder / f"{job.stem}.md"
        md_path.write_text(markdown, encoding="utf-8")
        job.outputs["md"] = md_path.name
        job.add_log(f"Markdown saved: output/{job.stem}/{md_path.name}")

        # 3. markdown -> pdf
        if job.make_pdf:
            if job.cancelled():
                raise InterruptedError
            job.set_stage("pdf", "Rendering PDF")
            pdf_path = md_path.with_suffix(".pdf")
            pdf.md_to_pdf(md_path, pdf_path)
            job.outputs["pdf"] = pdf_path.name
            job.add_log(f"PDF saved: output/{job.stem}/{pdf_path.name}")

        job.progress, job.status, job.message = 1.0, "done", "Finished"
    except InterruptedError:
        job.status, job.message = "cancelled", "Cancelled"
        job.add_log("Cancelled by user.")
    except Exception as exc:
        job.status, job.message, job.error = "error", "Failed", str(exc) or exc.__class__.__name__
        job.add_log(f"ERROR: {job.error}")
        if not isinstance(exc, composer.ComposeError):
            traceback.print_exc()
        if job.outputs.get("transcript") and "md" not in job.outputs:
            job.add_log(f"The transcript was kept — upload output/{job.stem}/{TRANSCRIPT_NAME} "
                        "to retry without transcribing again.")
    finally:
        job.finished = time.time()
        transcribe.unload()
        for src in job.sources:
            try:
                src.unlink(missing_ok=True)
            except OSError:
                pass
        try:
            job.folder.rmdir()  # only succeeds when the session produced nothing
        except OSError:
            pass
