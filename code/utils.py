from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

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

