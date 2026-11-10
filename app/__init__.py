from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STATIC_DIR = Path(__file__).resolve().parent / "static"
OUTPUT_DIR = ROOT / "output"
TRANSCRIPT_DIR = OUTPUT_DIR / "transcripts"
UPLOAD_DIR = ROOT / "uploads"

for _d in (OUTPUT_DIR, TRANSCRIPT_DIR, UPLOAD_DIR):
    _d.mkdir(parents=True, exist_ok=True)
