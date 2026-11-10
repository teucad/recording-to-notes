from __future__ import annotations

import os
import subprocess
import uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from . import OUTPUT_DIR, STATIC_DIR, UPLOAD_DIR
from . import compose as composer
from . import jobs, pdf, transcribe

AUDIO_EXTS = {".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg", ".opus", ".wma", ".webm", ".mp4", ".mkv", ".mov"}

app = FastAPI(title="Voice to Notes")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return (STATIC_DIR / "index.html").read_text(encoding="utf-8")


def _gpu() -> dict | None:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        ).stdout.strip().splitlines()[0]
        name, total, used = (s.strip() for s in out.split(","))
        return {"name": name, "vram_total_mb": int(total), "vram_used_mb": int(used)}
    except Exception:
        return None


@app.get("/api/system")
def system() -> dict:
    models = composer.list_models()
    active = jobs.active_job()
    return {
        "gpu": _gpu(),
        "cuda": transcribe.cuda_available(),
        "ollama": models is not None,
        "llm_models": models or [],
        "whisper_models": transcribe.WHISPER_MODELS,
        "katex": (STATIC_DIR / "vendor" / "katex" / "katex.min.js").exists(),
        "output_dir": str(OUTPUT_DIR),
        "active_job": active.id if active else None,
    }


async def _save_upload(file: UploadFile, dest: Path) -> None:
    with dest.open("wb") as fh:
        while chunk := await file.read(1 << 20):
            fh.write(chunk)


@app.post("/api/jobs")
async def create_job(
    file: UploadFile = File(...),
    title: str = Form(""),
    language: str = Form(""),
    out_language: str = Form(""),
    whisper_model: str = Form("large-v3-turbo"),
    llm_model: str = Form(""),
    make_pdf: bool = Form(True),
) -> dict:
    name = Path(file.filename or "lecture").name
    ext = Path(name).suffix.lower()
    if ext not in AUDIO_EXTS | jobs.TEXT_EXTS:
        raise HTTPException(400, f"Unsupported file type '{ext}'. Use audio/video or a .txt transcript.")
    if whisper_model not in transcribe.WHISPER_MODELS:
        raise HTTPException(400, "Unknown Whisper model.")
    if jobs.active_job():
        raise HTTPException(409, "Another lecture is already being processed.")
    source = UPLOAD_DIR / f"{uuid.uuid4().hex}{ext}"
    await _save_upload(file, source)
    job = jobs.Job(
        source=source, source_name=name, stem=jobs.unique_stem(title.strip() or Path(name).stem),
        title=title.strip(), language=language.strip(), out_language=out_language.strip(),
        whisper_model=whisper_model, llm_model=llm_model.strip(), make_pdf=make_pdf,
    )
    try:
        jobs.start(job)
    except RuntimeError as exc:
        source.unlink(missing_ok=True)
        raise HTTPException(409, str(exc))
    return job.to_dict()


def _job(job_id: str) -> jobs.Job:
    job = jobs.JOBS.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found.")
    return job


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> dict:
    return _job(job_id).to_dict()


@app.post("/api/jobs/{job_id}/cancel")
def cancel_job(job_id: str) -> dict:
    job = _job(job_id)
    job.cancel_flag = True
    return job.to_dict()


def _output_file(name: str, exts: set[str]) -> Path:
    if Path(name).name != name or Path(name).suffix.lower() not in exts:
        raise HTTPException(400, "Invalid file name.")
    path = OUTPUT_DIR / name
    if not path.is_file():
        raise HTTPException(404, "File not found.")
    return path


@app.get("/api/files")
def list_files() -> list[dict]:
    items = []
    for md in OUTPUT_DIR.glob("*.md"):
        pdf_path = md.with_suffix(".pdf")
        has_pdf = pdf_path.exists()
        items.append({
            "stem": md.stem,
            "title": pdf.title_of(md.read_text(encoding="utf-8", errors="replace")[:2000], md.stem),
            "md": md.name,
            "pdf": pdf_path.name if has_pdf else None,
            "pdf_stale": has_pdf and pdf_path.stat().st_mtime < md.stat().st_mtime,
            "modified": md.stat().st_mtime,
            "words": len(md.read_text(encoding="utf-8", errors="replace").split()),
        })
    return sorted(items, key=lambda i: i["modified"], reverse=True)


@app.get("/api/files/{name}")
def get_file(name: str, download: bool = False) -> FileResponse:
    path = _output_file(name, {".md", ".pdf"})
    media = "application/pdf" if path.suffix == ".pdf" else "text/markdown; charset=utf-8"
    return FileResponse(path, media_type=media, filename=path.name,
                        content_disposition_type="attachment" if download else "inline")


@app.get("/api/preview/{name}", response_class=HTMLResponse)
def preview(name: str) -> str:
    path = _output_file(name, {".md"})
    text = path.read_text(encoding="utf-8", errors="replace")
    return pdf.build_html(text, pdf.title_of(text, path.stem), "/static")


@app.post("/api/convert/{name}")
def convert_existing(name: str) -> dict:
    """Convert a Markdown file that is already in the output folder to PDF."""
    path = _output_file(name, {".md"})
    out = path.with_suffix(".pdf")
    try:
        pdf.md_to_pdf(path, out)
    except Exception as exc:
        raise HTTPException(500, f"PDF conversion failed: {exc}")
    return {"md": path.name, "pdf": out.name}


@app.post("/api/convert-upload")
async def convert_upload(file: UploadFile = File(...)) -> dict:
    """Copy any Markdown file into the output folder and convert it to PDF."""
    name = Path(file.filename or "notes.md").name
    if Path(name).suffix.lower() not in {".md", ".markdown"}:
        raise HTTPException(400, "Please choose a .md file.")
    md_path = OUTPUT_DIR / f"{jobs.unique_stem(Path(name).stem)}.md"
    await _save_upload(file, md_path)
    return await _in_thread(convert_existing, md_path.name)


async def _in_thread(fn, *args):
    from starlette.concurrency import run_in_threadpool

    return await run_in_threadpool(fn, *args)


@app.post("/api/open-output")
def open_output() -> dict:
    if os.name == "nt":
        os.startfile(OUTPUT_DIR)  # noqa: S606 - opens the fixed output folder in Explorer
    return {"path": str(OUTPUT_DIR)}
