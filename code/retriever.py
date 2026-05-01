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


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"


ALLOWED_EXTS = {".txt", ".md", ".html", ".json", ".csv"}


def _safe_read_text(path: Path) -> str:
    # Be resilient to weird encodings in corpora.
    return path.read_text(encoding="utf-8", errors="ignore")


def _infer_source_company_from_path(path: Path) -> str:
    # Determine source by the *first directory under data/*.
    # IMPORTANT: do not substring-match the repo root name "hackerrank", or
    # everything will be mislabeled on Windows because the repo folder is
    # ...\Desktop\hackerrank\...
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
        self.model_name = model_name
        self.chunk_chars = chunk_chars
        self.overlap_chars = overlap_chars
        self.min_score = min_score

        self._model = None
        self._docs: list[CorpusDoc] = []
        self._chunks: list[Chunk] = []
        self._embeddings: np.ndarray | None = None

    def load_and_index(self) -> None:
        debug_log(
            run_id=os.getenv("DEBUG_RUN_ID", "pre-fix"),
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
            run_id=os.getenv("DEBUG_RUN_ID", "pre-fix"),
            hypothesis_id="H6",
            location="retriever.py:load_and_index",
            message="Corpus docs by source_company",
            data={"counts": counts, "total": len(self._docs)},
        )
        self._chunks = self.chunk_documents(self._docs)
        self._embeddings = self.build_index(self._chunks)
        debug_log(
            run_id=os.getenv("DEBUG_RUN_ID", "pre-fix"),
            hypothesis_id="H1",
            location="retriever.py:load_and_index",
            message="Finished corpus load/index",
            data={"num_docs": len(self._docs), "num_chunks": len(self._chunks), "embeddings": None if self._embeddings is None else list(self._embeddings.shape)},
        )

    def load_corpus(self) -> list[CorpusDoc]:
        docs: list[CorpusDoc] = []
        if not DATA_DIR.exists():
            return docs

        for path in DATA_DIR.rglob("*"):
            if not path.is_file():
                continue
            if path.suffix.lower() not in ALLOWED_EXTS:
                continue
            # Skip embedding cache artifacts (not part of support corpus)
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

        step = max(1, self.chunk_chars - self.overlap_chars)
        for d in docs:
            text = d.text
            n = len(text)
            if n <= self.chunk_chars:
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

            for start in range(0, n, step):
                end = min(n, start + self.chunk_chars)
                chunk_text = text[start:end]
                if chunk_text.strip():
                    chunks.append(
                        Chunk(
                            text=chunk_text,
                            chunk_id=chunk_id,
                            source_company=d.source_company,
                            filename=d.filename,
                            char_start=start,
                        )
                    )
                    chunk_id += 1
                if end >= n:
                    break

        return chunks

    def _cache_paths(self) -> tuple[Path, Path]:
        # Match plan.md cache location: data/embeddings_cache.npy
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
                            run_id=os.getenv("DEBUG_RUN_ID", "pre-fix"),
                            hypothesis_id="H1",
                            location="retriever.py:build_index",
                            message="Embedding cache hit",
                            data={"emb_path": str(emb_path), "rows": int(emb.shape[0]), "dim": int(emb.shape[1]) if emb.ndim == 2 else None},
                        )
                        return emb
            except Exception:
                pass
        debug_log(
            run_id=os.getenv("DEBUG_RUN_ID", "pre-fix"),
            hypothesis_id="H1",
            location="retriever.py:build_index",
            message="Embedding cache miss; recomputing",
            data={"emb_path": str(emb_path), "meta_path": str(meta_path), "num_chunks": len(chunks)},
        )

        if SentenceTransformer is None:
            raise RuntimeError(
                "sentence-transformers is required for retrieval. "
                "Install dependencies: pip install -r requirements.txt"
            )

        self._model = SentenceTransformer(self.model_name)
        texts = [c.text for c in chunks]

        emb = self._model.encode(texts, batch_size=64, show_progress_bar=True, normalize_embeddings=True)
        emb = np.asarray(emb, dtype=np.float32)

        np.save(emb_path, emb)
        meta_path.write_text(json.dumps(meta_key), encoding="utf-8")

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

        sims = cosine_similarity(q_emb, self._embeddings)[0]  # shape: (num_chunks,)
        sims = sims.astype(np.float32, copy=False)

        if company is not None:
            c = company.strip().lower()
            for i, ch in enumerate(self._chunks):
                if ch.source_company.lower() == c:
                    sims[i] *= 1.5

        # Get top candidates
        k = min(max(1, top_k * 5), len(self._chunks))
        top_idx = np.argpartition(-sims, kth=k - 1)[:k]
        # Sort by score desc
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

        if results:
            debug_log(
                run_id=os.getenv("DEBUG_RUN_ID", "pre-fix"),
                hypothesis_id="H2",
                location="retriever.py:retrieve",
                message="Retrieved chunks",
                data={
                    "company_boost": company,
                    "top_score": float(results[0]["score"]),
                    "kept": len(results),
                    "top_source": results[0].get("source_company"),
                    "top_file": results[0].get("filename"),
                    "sources": [r.get("source_company") for r in results],
                },
            )
        else:
            debug_log(
                run_id=os.getenv("DEBUG_RUN_ID", "pre-fix"),
                hypothesis_id="H2",
                location="retriever.py:retrieve",
                message="No chunks above threshold",
                data={"company_boost": company, "min_score": self.min_score},
            )

        return results

