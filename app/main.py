from __future__ import annotations

import os
import subprocess
import uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool

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
        "languages": sorted(composer.LANG_NAMES.items(), key=lambda kv: kv[1]),
        "katex": (STATIC_DIR / "vendor" / "katex" / "katex.min.js").exists(),
        "output_dir": str(OUTPUT_DIR),
        "active_job": active.id if active else None,
    }


async def _save_upload(file: UploadFile, dest: Path) -> None:
    with dest.open("wb") as fh:
        while chunk := await file.read(1 << 20):
            fh.write(chunk)


# ---------- sessions ----------

@app.post("/api/jobs")
async def create_job(
    files: list[UploadFile] = File(...),
    title: str = Form(""),
    language: str = Form(""),
    out_language: str = Form(""),
    whisper_model: str = Form("large-v3-turbo"),
    llm_model: str = Form(""),
    make_pdf: bool = Form(True),
) -> dict:
    """Start a session. `files` are processed in the order they are sent."""
    names = [Path(f.filename or "lecture").name for f in files]
    for name in names:
        ext = Path(name).suffix.lower()
        if ext not in AUDIO_EXTS | jobs.TEXT_EXTS:
            raise HTTPException(400, f"Unsupported file type '{ext}' ({name}). Use audio/video or a .txt transcript.")
    if whisper_model not in transcribe.WHISPER_MODELS:
        raise HTTPException(400, "Unknown Whisper model.")
    if jobs.active_job():
        raise HTTPException(409, "Another session is already being processed.")

    sources: list[Path] = []
    for f, name in zip(files, names):
        dest = UPLOAD_DIR / f"{uuid.uuid4().hex}{Path(name).suffix.lower()}"
        await _save_upload(f, dest)
        sources.append(dest)
    job = jobs.Job(
        sources=sources, source_names=names,
        stem=jobs.unique_stem(title.strip() or Path(names[0]).stem),
        title=title.strip(), language=language.strip(), out_language=out_language.strip(),
        whisper_model=whisper_model, llm_model=llm_model.strip(), make_pdf=make_pdf,
    )
    try:
        jobs.start(job)
    except RuntimeError as exc:
        for s in sources:
            s.unlink(missing_ok=True)
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


# ---------- output folder: output/<session>/<files> ----------

def _session_dir(folder: str) -> Path:
    if Path(folder).name != folder or folder in ("", ".", ".."):
        raise HTTPException(400, "Invalid folder name.")
    path = OUTPUT_DIR / folder
    if not path.is_dir():
        raise HTTPException(404, "Folder not found.")
    return path


def _output_file(folder: str, name: str, exts: set[str]) -> Path:
    if Path(name).name != name or Path(name).suffix.lower() not in exts:
        raise HTTPException(400, "Invalid file name.")
    path = _session_dir(folder) / name
    if not path.is_file():
        raise HTTPException(404, "File not found.")
    return path


@app.get("/api/files")
def list_files() -> list[dict]:
    items = []
    for folder in (d for d in OUTPUT_DIR.iterdir() if d.is_dir()):
        for md in folder.glob("*.md"):
            text = md.read_text(encoding="utf-8", errors="replace")
            pdf_path = md.with_suffix(".pdf")
            has_pdf = pdf_path.exists()
            items.append({
                "folder": folder.name,
                "title": pdf.title_of(text[:2000], md.stem),
                "md": md.name,
                "pdf": pdf_path.name if has_pdf else None,
                "pdf_stale": has_pdf and pdf_path.stat().st_mtime < md.stat().st_mtime,
                "transcript": (folder / jobs.TRANSCRIPT_NAME).exists(),
                "modified": md.stat().st_mtime,
                "words": len(text.split()),
            })
    return sorted(items, key=lambda i: i["modified"], reverse=True)


@app.get("/api/files/{folder}/{name}")
def get_file(folder: str, name: str, download: bool = False) -> FileResponse:
    path = _output_file(folder, name, {".md", ".pdf", ".txt"})
    media = {".pdf": "application/pdf", ".md": "text/markdown; charset=utf-8"}.get(path.suffix, "text/plain; charset=utf-8")
    return FileResponse(path, media_type=media, filename=path.name,
                        content_disposition_type="attachment" if download else "inline")


@app.get("/api/preview/{folder}/{name}", response_class=HTMLResponse)
def preview(folder: str, name: str) -> str:
    path = _output_file(folder, name, {".md"})
    text = path.read_text(encoding="utf-8", errors="replace")
    return pdf.build_html(text, pdf.title_of(text, path.stem), "/static")


@app.post("/api/convert/{folder}/{name}")
def convert_existing(folder: str, name: str) -> dict:
    """Convert a Markdown file that is already in the output folder to PDF."""
    path = _output_file(folder, name, {".md"})
    out = path.with_suffix(".pdf")
    try:
        pdf.md_to_pdf(path, out)
    except Exception as exc:
        raise HTTPException(500, f"PDF conversion failed: {exc}")
    return {"folder": folder, "md": path.name, "pdf": out.name}


@app.post("/api/convert-upload")
async def convert_upload(file: UploadFile = File(...)) -> dict:
    """Copy any Markdown file into its own output folder and convert it to PDF."""
    name = Path(file.filename or "notes.md").name
    if Path(name).suffix.lower() not in {".md", ".markdown"}:
        raise HTTPException(400, "Please choose a .md file.")
    stem = jobs.unique_stem(Path(name).stem)
    (OUTPUT_DIR / stem).mkdir(parents=True)
    await _save_upload(file, OUTPUT_DIR / stem / f"{stem}.md")
    return await run_in_threadpool(convert_existing, stem, f"{stem}.md")


@app.post("/api/open-output")
def open_output(folder: str = "") -> dict:
    """Open the output folder (or one session folder) in Explorer."""
    path = _session_dir(folder) if folder else OUTPUT_DIR
    if os.name == "nt":
        os.startfile(path)  # noqa: S606 - only ever a folder inside output/
    return {"path": str(path)}
