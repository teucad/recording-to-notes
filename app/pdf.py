"""Markdown -> styled HTML -> PDF (headless Chromium via Playwright)."""
from __future__ import annotations

import html
import re
import tempfile
from pathlib import Path

from markdown_it import MarkdownIt
from mdit_py_plugins.anchors import anchors_plugin
from mdit_py_plugins.deflist import deflist_plugin
from mdit_py_plugins.dollarmath import dollarmath_plugin
from mdit_py_plugins.footnote import footnote_plugin
from mdit_py_plugins.tasklists import tasklists_plugin

from . import STATIC_DIR

_md = (
    MarkdownIt("commonmark", {"html": False, "typographer": True})
    .enable(["table", "strikethrough", "replacements", "smartquotes"])
    .use(dollarmath_plugin, allow_space=True, double_inline=True)
    .use(footnote_plugin)
    .use(deflist_plugin)
    .use(tasklists_plugin)
    .use(anchors_plugin, min_level=2, max_level=3)
)


def _normalise_math(text: str) -> str:
    r"""Accept \( \) and \[ \] delimiters too (outside fenced code)."""
    parts = re.split(r"(```.*?```)", text, flags=re.S)
    for i in range(0, len(parts), 2):
        parts[i] = re.sub(r"\\\[(.+?)\\\]", lambda m: f"\n$$\n{m.group(1).strip()}\n$$\n", parts[i], flags=re.S)
        parts[i] = re.sub(r"\\\((.+?)\\\)", lambda m: f"${m.group(1).strip()}$", parts[i])
    return "".join(parts)


def _toc(tokens) -> str:
    items = []
    for i, tok in enumerate(tokens):
        if tok.type == "heading_open" and tok.tag in ("h2", "h3") and tok.attrGet("id"):
            inline = tokens[i + 1]
            label = "".join(c.content for c in (inline.children or []) if c.type in ("text", "code_inline"))
            items.append((tok.tag, tok.attrGet("id"), label or inline.content))
    if sum(1 for t, _, _ in items if t == "h2") < 3:
        return ""
    lis = "".join(
        f'<li class="toc-{tag}"><a href="#{html.escape(str(hid))}">{html.escape(label)}</a></li>'
        for tag, hid, label in items
    )
    return f'<nav class="toc"><div class="toc-title">Contents</div><ul>{lis}</ul></nav>'


def title_of(md_text: str, fallback: str) -> str:
    m = re.search(r"(?m)^# (.+)$", md_text)
    return m.group(1).strip() if m else fallback


def build_html(md_text: str, title: str, asset_base: str) -> str:
    """Full HTML document. `asset_base` is a URL prefix for the static assets."""
    tokens = _md.parse(_normalise_math(md_text))
    body = _md.renderer.render(tokens, _md.options, {})
    toc = _toc(tokens)
    if toc:
        body = body.replace("<h2", toc + "<h2", 1)
    return f"""<!doctype html>
<html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title>
<link rel="stylesheet" href="{asset_base}/vendor/katex/katex.min.css">
<link rel="stylesheet" href="{asset_base}/textbook.css">
</head><body><main class="book">
{body}
</main>
<script src="{asset_base}/vendor/katex/katex.min.js"></script>
<script>
try {{
  document.querySelectorAll('.math').forEach(function (el) {{
    katex.render(el.textContent, el, {{displayMode: el.classList.contains('block'), throwOnError: false}});
  }});
}} catch (e) {{}}
(document.fonts ? document.fonts.ready : Promise.resolve()).then(function () {{ window.__rendered = true; }});
</script>
</body></html>"""


def md_to_pdf(md_path: Path, pdf_path: Path) -> None:
    from playwright.sync_api import sync_playwright

    md_text = md_path.read_text(encoding="utf-8")
    title = title_of(md_text, md_path.stem)
    doc = build_html(md_text, title, STATIC_DIR.as_uri())
    small = "font-family:Georgia,serif;font-size:8px;color:#777;width:100%;padding:0 18mm;"
    with tempfile.TemporaryDirectory() as tmp:
        page_file = Path(tmp) / "page.html"
        page_file.write_text(doc, encoding="utf-8")
        with sync_playwright() as p:
            browser = None
            errors = []
            for kwargs in ({}, {"channel": "msedge"}, {"channel": "chrome"}):
                try:
                    browser = p.chromium.launch(**kwargs)
                    break
                except Exception as exc:
                    errors.append(str(exc).splitlines()[0])
            if browser is None:
                raise RuntimeError(
                    "No headless browser available for PDF export. Run: uv run playwright install chromium --only-shell"
                    f" ({errors[0]})"
                )
            try:
                page = browser.new_page()
                page.goto(page_file.as_uri())
                page.wait_for_function("window.__rendered === true", timeout=30000)
                page.pdf(
                    path=str(pdf_path),
                    format="A4",
                    print_background=True,
                    margin={"top": "22mm", "bottom": "22mm", "left": "20mm", "right": "20mm"},
                    display_header_footer=True,
                    header_template=f'<div style="{small}text-align:right;">{html.escape(title)}</div>',
                    footer_template=f'<div style="{small}text-align:center;"><span class="pageNumber"></span></div>',
                )
            finally:
                browser.close()
