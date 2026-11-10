# One-time setup: Python environment, CUDA libraries, headless browser for PDF, KaTeX.
$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot

uv sync --python 3.12
if (-not $?) { throw 'uv sync failed' }
uv run playwright install chromium --only-shell
if (-not $?) { throw 'playwright install failed' }
uv run python scripts/fetch_katex.py
if (-not $?) { throw 'KaTeX download failed' }

Write-Host "`nSetup complete. Start the app with run.bat"
