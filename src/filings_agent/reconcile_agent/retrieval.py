"""Retrieval over 10-K filings: the RAG behind the agent's `search_filing` tool.

Pipeline (LangChain primitives, per the docs' agentic-RAG pattern):
  filing markdown -> RecursiveCharacterTextSplitter -> drop table-only chunks
  -> embeddings (Bedrock Titan v2, cached on disk per chunk) -> InMemoryVectorStore per filing

Two search modes, chosen in config (`retrieval.mode`), so they can be compared in evals:
  vector  similarity search only
  hybrid  vector + keyword ranking merged by reciprocal rank fusion (RRF). Keyword search catches
          exact terms embeddings blur ("ten-for-one", "ASU 2018-12", a spun-off company's name).

Embeddings are the only paid part and each chunk is embedded once: the cache key is the chunk
text + model, so re-indexing, re-running evals and new search modes cost nothing.
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
from collections import defaultdict
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

from filings_agent.fsutil import atomic_write_text
from filings_agent.textnorm import normalize


class CachedEmbeddings(Embeddings):
    """Wraps any LangChain Embeddings with a per-text disk cache and parallel misses."""

    def __init__(self, base: Embeddings, cache_dir: Path, model_id: str, concurrency: int = 8):
        self._base = base
        self._dir = cache_dir
        self._model = model_id
        self._concurrency = concurrency

    def _path(self, text: str) -> Path:
        key = hashlib.sha256(f"{self._model}|{text}".encode()).hexdigest()
        return self._dir / key[:2] / f"{key}.json"

    def _one(self, text: str) -> list[float]:
        path = self._path(text)
        if path.exists():
            return cast(list[float], json.loads(path.read_text()))
        vec = self._base.embed_query(text)
        atomic_write_text(path, json.dumps(vec))
        return vec

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        # Dedupe first: identical passages in one batch would otherwise all miss the cache at
        # once and each be embedded (and paid for).
        unique = list(dict.fromkeys(texts))
        with ThreadPoolExecutor(self._concurrency) as pool:
            vectors = dict(zip(unique, pool.map(self._one, unique), strict=True))
        return [vectors[t] for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._one(text)


@dataclass(frozen=True)
class Passage:
    passage_id: str  # "<accession>#<n>": cited by the agent, checked by evals
    accession_no: str
    text: str
    score: float


_WORD = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")


def _tokens(text: str) -> list[str]:
    return _WORD.findall(normalize(text))


def is_table_chunk(text: str) -> bool:
    """Numeric tables have many pipes and few words; the explanations we need are prose."""
    letters = sum(c.isalpha() for c in text)
    return text.count("|") > 20 or letters < 0.4 * max(len(text), 1)


def chunk_filing(markdown: str, accession_no: str, size: int, overlap: int) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=size, chunk_overlap=overlap)
    docs: list[Document] = []
    for i, text in enumerate(splitter.split_text(markdown)):
        clean = re.sub(r"\s+", " ", text).strip()
        if not clean or is_table_chunk(clean):
            continue
        docs.append(
            Document(
                page_content=clean,
                id=f"{accession_no}#{i}",
                metadata={"accession_no": accession_no, "chunk": i},
            )
        )
    return docs


class FilingIndex:
    """Lazily builds one vector store per filing; memoized for the life of the process."""

    def __init__(
        self,
        markdown_of: Callable[[str], str],
        embeddings: Embeddings,
        chunk_size: int = 1500,
        chunk_overlap: int = 200,
        mode: str = "vector",
        extra_chunks: dict[str, list[str]] | None = None,  # eval-only: injected (synthetic) text
    ) -> None:
        self._markdown_of = markdown_of
        self._embeddings = embeddings
        self._size = chunk_size
        self._overlap = chunk_overlap
        self.mode = mode
        self._extra = extra_chunks or {}
        self._stores: dict[str, tuple[InMemoryVectorStore, list[Document]]] = {}
        self._locks: dict[str, threading.Lock] = defaultdict(threading.Lock)
        self._locks_guard = threading.Lock()

    def with_extra_chunks(self, extra: dict[str, list[str]]) -> FilingIndex:
        """A copy whose filings also contain `extra` passages (eval-only: synthetic injections).
        Shares the embedding cache, so only the new passages are embedded."""
        return FilingIndex(
            self._markdown_of, self._embeddings, self._size, self._overlap, self.mode, extra
        )

    @property
    def version(self) -> str:
        """Part of the agent's cache key: a retrieval change must invalidate cached answers."""
        extra = hashlib.sha256(json.dumps(self._extra, sort_keys=True).encode()).hexdigest()[:8]
        return f"chunk{self._size}-{self._overlap}/{self.mode}/x{extra}"

    def _lock(self, accession_no: str) -> threading.Lock:
        with self._locks_guard:
            return self._locks[accession_no]

    def warm(self, accession_nos: list[str], concurrency: int = 4) -> int:
        """Build indexes up front (parallel across filings) so evals don't pay indexing inside
        agent tool calls. Returns the number of passages indexed."""
        with ThreadPoolExecutor(concurrency) as pool:
            return sum(len(docs) for _, docs in pool.map(self._store, accession_nos))

    def _store(self, accession_no: str) -> tuple[InMemoryVectorStore, list[Document]]:
        with self._lock(accession_no):  # two agents asking for the same filing build it once
            return self._build(accession_no)

    def _build(self, accession_no: str) -> tuple[InMemoryVectorStore, list[Document]]:
        if accession_no not in self._stores:
            docs = chunk_filing(
                self._markdown_of(accession_no), accession_no, self._size, self._overlap
            )
            for j, text in enumerate(self._extra.get(accession_no, [])):
                docs.append(
                    Document(
                        page_content=text,
                        id=f"{accession_no}#x{j}",
                        metadata={"accession_no": accession_no, "chunk": -1},
                    )
                )
            store = InMemoryVectorStore(self._embeddings)
            store.add_documents(docs)
            self._stores[accession_no] = (store, docs)
        return self._stores[accession_no]

    def search(self, query: str, accession_no: str, k: int = 5) -> list[Passage]:
        store, docs = self._store(accession_no)
        vector = [(d, s) for d, s in store.similarity_search_with_score(query, k=max(k, 20))]
        if self.mode == "vector":
            return [_passage(d, s) for d, s in vector[:k]]
        return _rrf(vector, _keyword_rank(query, docs), k)


def _passage(doc: Document, score: float) -> Passage:
    return Passage(str(doc.id), str(doc.metadata["accession_no"]), doc.page_content, float(score))


def _keyword_rank(query: str, docs: list[Document]) -> list[Document]:
    """Simple keyword relevance: query-term coverage, rarer terms weighted higher."""
    q = set(_tokens(query))
    if not q:
        return []
    doc_tokens = [set(_tokens(d.page_content)) for d in docs]
    df = {t: sum(1 for toks in doc_tokens if t in toks) for t in q}
    scored = [
        (sum(1.0 / df[t] for t in q if t in toks and df[t]), d)
        for d, toks in zip(docs, doc_tokens, strict=True)
    ]
    return [d for s, d in sorted(scored, key=lambda x: -x[0]) if s > 0][:20]


def _rrf(
    vector: list[tuple[Document, float]], keyword: list[Document], k: int, c: int = 60
) -> list[Passage]:
    """Reciprocal rank fusion: robust to the two rankers' scores being on different scales."""
    fused: dict[str, float] = {}
    by_id: dict[str, Document] = {}
    for rank, (d, _) in enumerate(vector):
        fused[str(d.id)] = fused.get(str(d.id), 0.0) + 1.0 / (c + rank + 1)
        by_id[str(d.id)] = d
    for rank, d in enumerate(keyword):
        fused[str(d.id)] = fused.get(str(d.id), 0.0) + 1.0 / (c + rank + 1)
        by_id[str(d.id)] = d
    best = sorted(fused.items(), key=lambda x: -x[1])[:k]
    return [_passage(by_id[i], s) for i, s in best]
