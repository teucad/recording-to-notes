"""The lecture pipeline: transcribe -> compose -> write Markdown -> export PDF."""
from __future__ import annotations

import re
import threading
import time
import traceback
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from . import OUTPUT_DIR, TRANSCRIPT_DIR
from . import compose as composer
from . import pdf, transcribe

# share of the overall progress bar per stage
WEIGHTS = {"transcribe": (0.0, 0.45), "compose": (0.45, 0.47), "pdf": (0.92, 0.08)}
TEXT_EXTS = {".txt"}


@dataclass
class Job:
    source: Path
    source_name: str
    stem: str
    title: str
    language: str          # spoken language code, "" = auto-detect
    out_language: str      # language of the notes, "" = same as the lecture
    whisper_model: str
    llm_model: str         # "" = no LLM, cleaned transcript only
    make_pdf: bool
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:10])
    status: str = "queued"  # queued | running | done | error | cancelled
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
            "id": self.id, "name": self.source_name, "status": self.status, "stage": self.stage,
            "progress": round(self.progress, 4), "message": self.message, "device": self.device,
            "error": self.error, "log": self.log[-200:], "outputs": self.outputs,
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
    base = slugify(name)
    stem, i = base, 2
    while any((OUTPUT_DIR / f"{stem}{ext}").exists() for ext in (".md", ".pdf")):
        stem = f"{base}-{i}"
        i += 1
    return stem


def start(job: Job) -> None:
    with _lock:
        if active_job():
            raise RuntimeError("Another lecture is already being processed.")
        JOBS[job.id] = job
    threading.Thread(target=_run, args=(job,), daemon=True).start()


def _run(job: Job) -> None:
    job.status = "running"
    try:
        # 1. speech -> text
        if job.source.suffix.lower() in TEXT_EXTS:
            job.set_stage("transcribe", "Reading transcript file")
            paragraphs = transcribe.paragraphs_from_text(job.source.read_text(encoding="utf-8", errors="replace"))
            detected = job.language
        else:
            job.set_stage("transcribe", "Transcribing audio")
            result = transcribe.transcribe(
                job.source, job.whisper_model, job.language,
                job.stage_progress, job.cancelled, job.add_log,
            )
            job.device = result["device"]
            paragraphs, detected = result["paragraphs"], result["language"]
            t_path = TRANSCRIPT_DIR / f"{job.stem}.txt"
            t_path.write_text(result["transcript_text"], encoding="utf-8")
            job.outputs["transcript"] = t_path.name
            job.add_log(f"Transcript saved: output/transcripts/{t_path.name}")
            transcribe.unload()  # hand the VRAM to the language model
        if not paragraphs:
            raise composer.ComposeError("No speech was found in this file.")

        # 2. text -> textbook chapter
        fallback_title = Path(job.source_name).stem.replace("_", " ").strip() or "Lecture"
        if job.llm_model:
            job.set_stage("compose", "Composing textbook chapter")
            markdown = composer.compose(
                paragraphs, job.llm_model, job.title, fallback_title, job.source_name,
                job.out_language or detected, job.stage_progress, job.cancelled, job.add_log,
            )
        else:
            job.set_stage("compose", "Formatting transcript (no language model selected)")
            markdown = composer.plain_markdown(paragraphs, job.title or fallback_title, job.source_name)
        md_path = OUTPUT_DIR / f"{job.stem}.md"
        md_path.write_text(markdown, encoding="utf-8")
        job.outputs["md"] = md_path.name
        job.add_log(f"Markdown saved: output/{md_path.name}")

        # 3. markdown -> pdf
        if job.make_pdf:
            if job.cancelled():
                raise InterruptedError
            job.set_stage("pdf", "Rendering PDF")
            pdf_path = OUTPUT_DIR / f"{job.stem}.pdf"
            pdf.md_to_pdf(md_path, pdf_path)
            job.outputs["pdf"] = pdf_path.name
            job.add_log(f"PDF saved: output/{pdf_path.name}")

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
            job.add_log("The transcript was kept — upload that .txt file to retry without transcribing again.")
    finally:
        job.finished = time.time()
        transcribe.unload()
        try:
            job.source.unlink(missing_ok=True)
        except OSError:
            pass
