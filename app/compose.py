"""Turn a raw lecture transcript into textbook-style Markdown with a local LLM (Ollama)."""
from __future__ import annotations

import json
import os
import re
import time
from datetime import date
from typing import Callable

import httpx

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")
CHUNK_WORDS = 900
CHUNK_CTX = 8192
FINAL_CTX = 16384
FINAL_BODY_WORDS = 4500

LANG_NAMES = {
    "en": "English", "tr": "Turkish", "de": "German", "fr": "French", "es": "Spanish",
    "it": "Italian", "pt": "Portuguese", "nl": "Dutch", "ru": "Russian", "ar": "Arabic",
    "zh": "Chinese", "ja": "Japanese", "ko": "Korean", "hi": "Hindi", "pl": "Polish",
    "uk": "Ukrainian", "el": "Greek", "sv": "Swedish", "da": "Danish", "no": "Norwegian",
    "fi": "Finnish", "cs": "Czech", "sk": "Slovak", "hu": "Hungarian", "ro": "Romanian",
    "bg": "Bulgarian", "sr": "Serbian", "hr": "Croatian", "bs": "Bosnian", "sl": "Slovenian",
    "mk": "Macedonian", "sq": "Albanian", "lt": "Lithuanian", "lv": "Latvian", "et": "Estonian",
    "ca": "Catalan", "gl": "Galician", "eu": "Basque", "he": "Hebrew", "fa": "Persian",
    "ur": "Urdu", "az": "Azerbaijani", "kk": "Kazakh", "uz": "Uzbek", "hy": "Armenian",
    "ka": "Georgian", "bn": "Bengali", "ta": "Tamil", "te": "Telugu", "mr": "Marathi",
    "gu": "Gujarati", "pa": "Punjabi", "ne": "Nepali", "id": "Indonesian", "ms": "Malay",
    "vi": "Vietnamese", "th": "Thai", "tl": "Filipino", "sw": "Swahili", "af": "Afrikaans",
}

SYSTEM = """You are an expert academic editor. You turn the raw transcript of a spoken lecture \
into a chapter of a well-written textbook.

Rules:
- Write the entire chapter in {lang}. The lecture may switch between languages; translate every passage that is in another language into {lang}. Where the lecturer uses an established technical term in another language, you may keep that original term in parentheses after its first use.
- Keep every substantive idea, definition, argument, example, number and formula from the transcript. \
Do not summarise away detail, and do not invent facts that the lecturer did not say.
- Rewrite spoken language as clear, formal, flowing textbook prose. Remove filler words, repetitions, \
false starts, greetings and classroom logistics. Silently correct obvious speech-recognition errors from context.
- Structure with Markdown: `##` for major sections and `###` for subsections, with descriptive titles. Never use `#`.
- Put a **term in bold** when it is first defined. Use bullet or numbered lists for enumerations and steps, \
and tables for comparisons, when that is clearer than prose.
- Set off important definitions, theorems and worked examples as block quotes that start with a bold label, \
e.g. `> **Definition.** ...` or `> **Example.** ...` (label translated into {lang}).
- Write mathematics in LaTeX: `$...$` inline and `$$...$$` on its own lines for displayed equations.
- Output only the Markdown of the chapter text. No preamble, no comments about the task, no code fence around the answer."""

CHUNK_PROMPT = """Lecture title: {title}
This is part {i} of {n} of the transcript.

Section headings already written in the chapter:
{headings}

The chapter currently ends with:
"...{tail}"

Transcript of part {i}:
\"\"\"
{chunk}
\"\"\"

Write the next portion of the chapter covering exactly this part of the transcript. \
If it continues the topic of the current section, continue directly with prose (or a `###` subsection) \
without repeating a heading; begin a new `##` section when the topic changes. {ending}"""

FINAL_PROMPT = """Below is a condensed version of a textbook chapter that was written from a lecture.

\"\"\"
{body}
\"\"\"

Write the front and back matter for this chapter in {lang}. Use exactly this layout, translating the \
five section headings into {lang}, and output nothing else:

TITLE: <a concise, descriptive chapter title>
=====FRONT=====
## Overview
<one paragraph of 80-150 words introducing what the chapter covers and why it matters>

### Learning Objectives
<3 to 6 bullet points, each starting with a verb>
=====BACK=====
## Key Takeaways
<5 to 8 bullet points>

## Glossary
<bullet list of the important terms, each as `- **term** — one-sentence definition`>

## Review Questions
<numbered list of 4 to 6 questions that test understanding>"""


class ComposeError(RuntimeError):
    pass


def list_models() -> list[str] | None:
    """Installed Ollama models, or None when Ollama is not reachable."""
    try:
        r = httpx.get(f"{OLLAMA_URL}/api/tags", timeout=3)
        r.raise_for_status()
        return sorted(m["name"] for m in r.json().get("models", []))
    except Exception:
        return None


def _chat(model, system, user, num_ctx, on_chars, should_cancel) -> str:
    payload = {
        "model": model,
        "stream": True,
        "think": False,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "options": {"num_ctx": num_ctx, "temperature": 0.3},
    }
    retries = [5, 15, 30]  # seconds to wait when the connection drops (e.g. Ollama restarting)
    drop_think = True
    while True:
        parts: list[str] = []
        try:
            with httpx.stream("POST", f"{OLLAMA_URL}/api/chat", json=payload,
                              timeout=httpx.Timeout(None, connect=10)) as r:
                if r.status_code >= 400:
                    detail = r.read().decode("utf-8", "replace")[:400]
                    if drop_think and "think" in detail.lower():
                        drop_think = False
                        payload.pop("think", None)  # model without a thinking mode
                        continue
                    raise ComposeError(f"Ollama returned {r.status_code}: {detail}")
                for line in r.iter_lines():
                    if should_cancel():
                        raise InterruptedError
                    if not line:
                        continue
                    msg = json.loads(line)
                    if msg.get("error"):
                        raise ComposeError(f"Ollama error: {msg['error']}")
                    piece = msg.get("message", {}).get("content", "")
                    if piece:
                        parts.append(piece)
                        on_chars(sum(map(len, parts)))
        except httpx.HTTPError as exc:
            if retries:
                wait = retries.pop(0)
                for _ in range(wait):
                    if should_cancel():
                        raise InterruptedError
                    time.sleep(1)
                continue
            raise ComposeError(
                f"Could not reach Ollama at {OLLAMA_URL} ({exc.__class__.__name__}). "
                "Start Ollama and try again."
            ) from exc
        return "".join(parts)


def _clean(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    text = text.strip()
    m = re.fullmatch(r"```(?:markdown|md)?\s*\n(.*)\n```", text, flags=re.S)
    if m:
        text = m.group(1).strip()
    text = re.sub(r"^# (?=\S)", "## ", text, flags=re.M)  # the chapter has a single H1
    return text


def _chunks(paragraphs: list[str]) -> list[str]:
    pieces: list[str] = []
    for p in paragraphs:  # split oversized paragraphs first
        words = p.split()
        for i in range(0, len(words), CHUNK_WORDS):
            pieces.append(" ".join(words[i:i + CHUNK_WORDS]))
    chunks: list[str] = []
    cur: list[str] = []
    count = 0
    for p in pieces:
        n = len(p.split())
        if cur and count + n > CHUNK_WORDS:
            chunks.append("\n\n".join(cur))
            cur, count = [], 0
        cur.append(p)
        count += n
    if cur:
        chunks.append("\n\n".join(cur))
    # avoid a tiny trailing chunk
    if len(chunks) > 1 and len(chunks[-1].split()) < CHUNK_WORDS // 4:
        chunks[-2] += "\n\n" + chunks.pop()
    return chunks


def _condense(body: str) -> str:
    """Trim every section evenly so the whole chapter fits the final prompt."""
    if len(body.split()) <= FINAL_BODY_WORDS:
        return body
    sections = re.split(r"(?m)^(?=## )", body)
    per = max(FINAL_BODY_WORDS // max(len(sections), 1), 60)
    out = []
    for s in sections:
        words = s.split(" ")
        out.append(" ".join(words[:per]) + (" …" if len(words) > per else ""))
    return "\n\n".join(out)


def _meta_line(source_name: str) -> str:
    return f"*Lecture notes · source: {source_name} · {date.today().isoformat()}*"


def plain_markdown(paragraphs: list[str], title: str, source_name: str) -> str:
    """Fallback when no language model is selected: a tidy transcript."""
    return f"# {title}\n\n{_meta_line(source_name)}\n\n## Transcript\n\n" + "\n\n".join(paragraphs) + "\n"


def compose(
    paragraphs: list[str],
    model: str,
    title: str,
    fallback_title: str,
    source_name: str,
    language: str,
    on_progress: Callable[[float, str], None],
    should_cancel: Callable[[], bool],
    log: Callable[[str], None],
) -> str:
    lang = LANG_NAMES.get(language, language) if language else "the same language as the transcript"
    system = SYSTEM.format(lang=lang)
    chunks = _chunks(paragraphs)
    n = len(chunks)
    if n == 0:
        raise ComposeError("The transcript is empty — no speech was detected.")
    log(f"Composing with {model}: {n} part(s), output language: {lang}.")

    body_parts: list[str] = []
    for i, chunk in enumerate(chunks, 1):
        body = "\n\n".join(body_parts)
        headings = "\n".join(re.findall(r"(?m)^#{2,3} .+$", body)) or "(none yet — this is the start of the chapter)"
        tail = " ".join(body.split()[-80:]) or "(nothing yet)"
        ending = (
            "This is the final part, so bring the chapter to a natural close, but do not add a summary section."
            if i == n else "Do not write a conclusion; the chapter continues."
        )
        expected = max(len(chunk), 1)

        def tick(chars: int, i=i, expected=expected) -> None:
            on_progress((i - 1 + min(chars / expected, 0.98)) / (n + 1), f"Writing part {i} of {n}")

        tick(0)
        text = _clean(_chat(
            model, system,
            CHUNK_PROMPT.format(title=title or "(untitled lecture)", i=i, n=n, headings=headings,
                                tail=tail, chunk=chunk, ending=ending),
            CHUNK_CTX, tick, should_cancel,
        ))
        if not text:
            raise ComposeError(f"The model returned no text for part {i}.")
        body_parts.append(text)
        log(f"Part {i}/{n} written ({len(text.split())} words).")

    body = "\n\n".join(body_parts)
    if not body.lstrip().startswith("## "):
        first = body.lstrip().split("\n", 1)[0]
        if not first.startswith("#"):
            body = "## Introduction\n\n" + body

    on_progress(n / (n + 1), "Writing overview, glossary and review questions")
    front = back = ""
    final_title = title
    try:
        raw = _clean(_chat(
            model, "You are an expert academic editor.",
            FINAL_PROMPT.format(body=_condense(body), lang=lang),
            FINAL_CTX, lambda c: None, should_cancel,
        ))
        m = re.search(r"(?m)^TITLE:\s*(.+)$", raw)
        if m and not final_title:
            final_title = m.group(1).strip().strip("*#\"' ")
        if "=====FRONT=====" in raw and "=====BACK=====" in raw:
            rest = raw.split("=====FRONT=====", 1)[1]
            front, back = (s.strip() for s in rest.split("=====BACK=====", 1))
        else:
            back = re.sub(r"(?m)^TITLE:.*$\n?", "", raw).replace("=====FRONT=====", "").replace("=====BACK=====", "").strip()
    except ComposeError as exc:
        log(f"Front/back matter skipped: {exc}")

    final_title = final_title or fallback_title
    parts = [f"# {final_title}", _meta_line(source_name)]
    if front:
        parts.append(front)
    parts.append(body)
    if back:
        parts.append(back)
    on_progress(1.0, "Chapter composed")
    return "\n\n".join(parts).strip() + "\n"
