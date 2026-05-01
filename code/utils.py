from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import re

import pandas as pd

# Debug-mode log file (NDJSON)
DEBUG_LOG_PATH = (Path(__file__).resolve().parents[1] / "debug-87b77d.log").resolve()
DEBUG_SESSION_ID = "87b77d"


ROOT_DIR = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class OutputRow:
    status: str
    product_area: str
    response: str
    justification: str
    request_type: str


OUTPUT_COLUMNS = ["status", "product_area", "response", "justification", "request_type"]


def load_tickets(csv_path: str | Path) -> pd.DataFrame:
    """
    Loads tickets CSV expected to have columns: Issue, Subject, Company.
    """
    p = Path(csv_path)
    if not p.is_absolute():
        p = ROOT_DIR / p

    df = pd.read_csv(p)

    # Normalize column names lightly (keep original casing in data frame)
    expected = {"Issue", "Subject", "Company"}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)} in {p}")

    # Ensure string columns (NaN -> empty string)
    for col in ["Issue", "Subject", "Company"]:
        df[col] = df[col].fillna("").astype(str)

    return df


def write_output(rows: Iterable[dict] | Iterable[OutputRow], output_path: str | Path) -> None:
    p = Path(output_path)
    if not p.is_absolute():
        p = ROOT_DIR / p

    normalized: list[dict] = []
    for r in rows:
        if isinstance(r, OutputRow):
            d = {
                "status": r.status,
                "product_area": r.product_area,
                "response": r.response,
                "justification": r.justification,
                "request_type": r.request_type,
            }
        else:
            d = dict(r)
        normalized.append(d)

    out_df = pd.DataFrame(normalized, columns=OUTPUT_COLUMNS)
    out_df.to_csv(p, index=False)


def normalize_company(raw: str) -> str | None:
    s = (raw or "").strip()
    if not s or s.lower() == "none":
        return None
    return s.strip().lower()


_MD_LINK_RE = re.compile(r"\[(.+?)\]\(.+?\)")
_MD_BOLD_RE = re.compile(r"\*\*(.+?)\*\*|__(.+?)__")
_MD_ITALIC_RE = re.compile(r"\*(.+?)\*|_(.+?)_")
_MD_CODE_BLOCK_RE = re.compile(r"```[\s\S]*?```")
_MD_INLINE_CODE_RE = re.compile(r"`([^`]+)`")
_MD_HEADING_RE = re.compile(r"^#{1,6}\s+", flags=re.MULTILINE)
_MD_BULLET_RE = re.compile(r"^[-*]\s+", flags=re.MULTILINE)
_MD_NUMBERED_RE = re.compile(r"^\d+\.\s+", flags=re.MULTILINE)
_HTML_TAG_RE = re.compile(r"<[^>]+>")


def sanitize_plaintext(text: str) -> str:
    """
    Strips common markdown and HTML formatting, then collapses excessive whitespace.
    Converts [label](url) to label.
    Removes bold/italic markers, code fences, backticks, headings, list bullets, numbered lists, and HTML tags.
    Collapses runs of blank lines to a single blank line.
    """
    if not text:
        return ""
    # Remove code blocks first (multiline)
    text = _MD_CODE_BLOCK_RE.sub("", text)
    # Convert inline code to plain text
    text = _MD_INLINE_CODE_RE.sub(r"\1", text)
    # Convert links to just the label
    text = _MD_LINK_RE.sub(r"\1", text)
    # Remove bold formatting (keep the content)
    text = _MD_BOLD_RE.sub(r"\1\2", text)
    # Remove italic formatting (keep the content)
    text = _MD_ITALIC_RE.sub(r"\1\2", text)
    # Remove headings
    text = _MD_HEADING_RE.sub("", text)
    # Remove list bullets
    text = _MD_BULLET_RE.sub("", text)
    # Remove numbered lists
    text = _MD_NUMBERED_RE.sub("", text)
    # Remove HTML tags
    text = _HTML_TAG_RE.sub("", text)
    # Collapse excessive blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def debug_log(
    *,
    run_id: str,
    hypothesis_id: str,
    location: str,
    message: str,
    data: dict | None = None,
) -> None:
    """
    Writes one NDJSON line for debug-mode analysis.
    Never log secrets (API keys, tokens, etc.).
    """
    #region agent log
    import json
    import time

    payload = {
        "sessionId": DEBUG_SESSION_ID,
        "runId": run_id,
        "hypothesisId": hypothesis_id,
        "location": location,
        "message": message,
        "data": data or {},
        "timestamp": int(time.time() * 1000),
    }
    try:
        # Ensure repo root exists (should), then append.
        DEBUG_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(DEBUG_LOG_PATH, "a", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception as e:
        # Last-resort visibility if file write fails (no secrets in payload).
        try:
            print(f"[debug_log] failed to write {DEBUG_LOG_PATH}: {e!r}")
        except Exception:
            pass
    #endregion agent log

