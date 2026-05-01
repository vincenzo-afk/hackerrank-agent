from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

from utils import debug_log

try:
    from sentence_transformers import SentenceTransformer
except Exception:  # pragma: no cover
    SentenceTransformer = None  # type: ignore

try:
    from bs4 import BeautifulSoup  # used for HTML stripping
    _BS4_AVAILABLE = True
except ImportError:
    _BS4_AVAILABLE = False

import re as _re

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"

ALLOWED_EXTS = {".txt", ".md", ".html", ".htm", ".json", ".csv"}


def _strip_html(raw: str) -> str:
    """Strip HTML tags and decode entities into plain text."""
    if _BS4_AVAILABLE:
        soup = BeautifulSoup(raw, "html.parser")
        # Remove script and style blocks entirely
        for tag in soup(["script", "style", "meta", "noscript", "head"]):
            tag.decompose()
        text = soup.get_text(separator=" ")
    else:
        # Fallback: regex strip
        text = _re.sub(r"<[^>]+>", " ", raw)
    # Collapse whitespace
    text = _re.sub(r"[ \t]+", " ", text)
    text = _re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _parse_json(raw: str) -> str:
    """
    Convert JSON to readable plain text.
    If the JSON is a list of objects, join them as key: value lines.
    Falls back to the raw text on parse failure.
    """
    try:
        obj = json.loads(raw)
    except Exception:
        return raw

    lines: list[str] = []

    def _flatten(o: object, prefix: str = "") -> None:
        if isinstance(o, dict):
            for k, v in o.items():
                _flatten(v, prefix=f"{prefix}{k}: " if not prefix else f"{prefix} > {k}: ")
        elif isinstance(o, list):
            for item in o:
                _flatten(item, prefix)
        else:
            lines.append(f"{prefix}{o}")

    _flatten(obj)
    return "\n".join(lines) if lines else raw


def _safe_read_text(path: Path) -> str:
    """Read a file and convert it to clean plain text based on its extension."""
    raw = path.read_text(encoding="utf-8", errors="ignore")
    ext = path.suffix.lower()
    if ext in {".html", ".htm"}:
        return _strip_html(raw)
    if ext == ".json":
        return _parse_json(raw)
    return raw


def _infer_source_company_from_path(path: Path) -> str:
    """
    Determine source company from the first directory under data/.
    IMPORTANT: do NOT substring-match on the repo root path (which might also
    contain "hackerrank" as the folder name on the developer's machine).
    """
    try:
        rel = path.relative_to(DATA_DIR)
        if rel.parts:
            top = rel.parts[0].lower()
            if top in {"hackerrank", "claude", "visa"}:
                return top
            return top
    except Exception:
        pass
    return "generic"


@dataclass(frozen=True)
class CorpusDoc:
    text: str
    source_company: str
    filename: str
    filepath: str


@dataclass(frozen=True)
class Chunk:
    text: str
    chunk_id: int
    source_company: str
    filename: str
    char_start: int


class CorpusRetriever:
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        chunk_chars: int = 1200,
        overlap_chars: int = 200,
        min_score: float = 0.25,
    ) -> None:
        self.model_name = os.getenv("EMBEDDING_MODEL", model_name)
        self.chunk_chars = int(os.getenv("CHUNK_CHARS", chunk_chars))
        self.overlap_chars = int(os.getenv("OVERLAP_CHARS", overlap_chars))
        self.min_score = float(os.getenv("MIN_SCORE", min_score))

        self._model = None
        self._docs: list[CorpusDoc] = []
        self._chunks: list[Chunk] = []
        self._embeddings: np.ndarray | None = None

    def load_and_index(self, force_reindex: bool = False) -> None:
        """
        Load corpus files, chunk them, and compute/load embeddings.
        """
        debug_log(
            run_id=os.getenv("DEBUG_RUN_ID", "run"),
            hypothesis_id="H1",
            location="retriever.py:load_and_index",
            message="Start corpus load/index",
            data={"model": self.model_name, "chunk_chars": self.chunk_chars, "overlap_chars": self.overlap_chars},
        )
        self._docs = self.load_corpus()

        # Source-company distribution sanity check
        counts: dict[str, int] = {}
        for d in self._docs:
            counts[d.source_company] = counts.get(d.source_company, 0) + 1
        debug_log(
            run_id=os.getenv("DEBUG_RUN_ID", "run"),
            hypothesis_id="H6",
            location="retriever.py:load_and_index",
            message="Corpus docs by source_company",
            data={"counts": counts, "total": len(self._docs)},
        )

        self._chunks = self.chunk_documents(self._docs)
        self._embeddings = self.build_index(self._chunks)
        debug_log(
            run_id=os.getenv("DEBUG_RUN_ID", "run"),
            hypothesis_id="H1",
            location="retriever.py:load_and_index",
            message="Finished corpus load/index",
            data={
                "num_docs": len(self._docs),
                "num_chunks": len(self._chunks),
                "embeddings_shape": None if self._embeddings is None else list(self._embeddings.shape),
            },
        )

    def load_corpus(self) -> list[CorpusDoc]:
        docs: list[CorpusDoc] = []
        if not DATA_DIR.exists():
            print(f"[WARNING] data/ directory not found at {DATA_DIR}")
            return docs

        for path in DATA_DIR.rglob("*"):
            if not path.is_file():
                continue
            if path.suffix.lower() not in ALLOWED_EXTS:
                continue
            # Skip embedding cache artifacts
            if path.name.startswith("embeddings_cache."):
                continue
            if path.parent.name.lower() == "cache":
                continue

            text = _safe_read_text(path)
            if not text or not text.strip():
                continue

            docs.append(
                CorpusDoc(
                    text=text,
                    source_company=_infer_source_company_from_path(path),
                    filename=path.name,
                    filepath=str(path),
                )
            )

        return docs

    def chunk_documents(self, docs: Iterable[CorpusDoc]) -> list[Chunk]:
        chunks: list[Chunk] = []
        chunk_id = 0
        
        for d in docs:
            text = d.text
            words = text.split()
            n_words = len(words)
            
            # Estimate words per chunk (~5 chars per word)
            words_per_chunk = max(10, self.chunk_chars // 5)
            words_overlap = max(5, self.overlap_chars // 5)
            step = max(1, words_per_chunk - words_overlap)

            if n_words <= words_per_chunk:
                chunks.append(
                    Chunk(
                        text=text,
                        chunk_id=chunk_id,
                        source_company=d.source_company,
                        filename=d.filename,
                        char_start=0,
                    )
                )
                chunk_id += 1
                continue

            for start_word in range(0, n_words, step):
                end_word = min(n_words, start_word + words_per_chunk)
                chunk_text = " ".join(words[start_word:end_word])
                if chunk_text.strip():
                    # Approximate char_start
                    char_start = sum(len(w) + 1 for w in words[:start_word])
                    chunks.append(
                        Chunk(
                            text=chunk_text,
                            chunk_id=chunk_id,
                            source_company=d.source_company,
                            filename=d.filename,
                            char_start=char_start,
                        )
                    )
                    chunk_id += 1
                if end_word >= n_words:
                    break

        return chunks

    def _cache_paths(self) -> tuple[Path, Path]:
        emb_path = DATA_DIR / "embeddings_cache.npy"
        meta_path = DATA_DIR / "embeddings_cache.meta.json"
        return emb_path, meta_path

    def build_index(self, chunks: list[Chunk]) -> np.ndarray:
        emb_path, meta_path = self._cache_paths()

        # Cache key: model + chunking params + corpus file list (paths + mtimes)
        corpus_fingerprint = []
        for d in self._docs:
            try:
                p = Path(d.filepath)
                corpus_fingerprint.append(
                    {"p": d.filepath, "m": p.stat().st_mtime_ns, "s": d.source_company}
                )
            except Exception:
                corpus_fingerprint.append({"p": d.filepath, "m": None, "s": d.source_company})

        meta_key = {
            "model": self.model_name,
            "chunk_chars": self.chunk_chars,
            "overlap_chars": self.overlap_chars,
            "docs": corpus_fingerprint,
            "num_chunks": len(chunks),
        }

        if emb_path.exists() and meta_path.exists():
            try:
                cached_meta = json.loads(meta_path.read_text(encoding="utf-8"))
                if cached_meta == meta_key:
                    emb = np.load(emb_path)
                    if isinstance(emb, np.ndarray) and emb.shape[0] == len(chunks):
                        debug_log(
                            run_id=os.getenv("DEBUG_RUN_ID", "run"),
                            hypothesis_id="H1",
                            location="retriever.py:build_index",
                            message="Embedding cache hit",
                            data={"rows": int(emb.shape[0])},
                        )
                        print(f"[retriever] Cache hit — loaded {emb.shape[0]} embeddings.")
                        return emb
            except Exception:
                pass

        print(f"[retriever] Building embedding index over {len(chunks)} chunks…")
        debug_log(
            run_id=os.getenv("DEBUG_RUN_ID", "run"),
            hypothesis_id="H1",
            location="retriever.py:build_index",
            message="Embedding cache miss; recomputing",
            data={"num_chunks": len(chunks)},
        )

        if SentenceTransformer is None:
            raise RuntimeError(
                "sentence-transformers is required. Run: pip install -r requirements.txt"
            )

        self._model = SentenceTransformer(self.model_name)
        texts = [c.text for c in chunks]
        emb = self._model.encode(texts, batch_size=64, show_progress_bar=True, normalize_embeddings=True)
        emb = np.asarray(emb, dtype=np.float32)

        np.save(emb_path, emb)
        meta_path.write_text(json.dumps(meta_key), encoding="utf-8")
        print(f"[retriever] Index built and cached to {emb_path}")

        return emb

    def retrieve(self, query: str, company: str | None = None, top_k: int = 5) -> list[dict]:
        if not query or not query.strip():
            return []
        if self._embeddings is None:
            raise RuntimeError("Retriever not indexed. Call load_and_index() first.")
        if not self._chunks:
            return []
        if SentenceTransformer is None:
            return []

        if self._model is None:
            self._model = SentenceTransformer(self.model_name)

        q_emb = self._model.encode([query], normalize_embeddings=True)
        q_emb = np.asarray(q_emb, dtype=np.float32)

        sims = cosine_similarity(q_emb, self._embeddings)[0]
        sims = sims.astype(np.float32, copy=False)

        # Boost chunks from the known company corpus
        if company is not None:
            c = company.strip().lower()
            for i, ch in enumerate(self._chunks):
                if ch.source_company.lower() == c:
                    sims[i] *= 1.5

        # Get top candidates
        k = min(max(1, top_k * 5), len(self._chunks))
        top_idx = np.argpartition(-sims, kth=k - 1)[:k]
        top_idx = top_idx[np.argsort(-sims[top_idx])]

        results: list[dict] = []
        for i in top_idx[:top_k]:
            score = float(sims[i])
            if math.isnan(score) or score < self.min_score:
                continue
            ch = self._chunks[int(i)]
            results.append(
                {
                    "text": ch.text,
                    "source_company": ch.source_company,
                    "filename": ch.filename,
                    "score": score,
                }
            )

        debug_log(
            run_id=os.getenv("DEBUG_RUN_ID", "run"),
            hypothesis_id="H2",
            location="retriever.py:retrieve",
            message="Retrieved chunks" if results else "No chunks above threshold",
            data={
                "company_boost": company,
                "top_score": float(results[0]["score"]) if results else None,
                "kept": len(results),
                "top_source": results[0].get("source_company") if results else None,
                "top_file": results[0].get("filename") if results else None,
                "sources": [r.get("source_company") for r in results],
            },
        )

        return results