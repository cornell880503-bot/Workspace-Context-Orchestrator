"""
Context Router & Ranker
  - Multi-query expansion
  - Hybrid search: BM25 (keyword) + ChromaDB (semantic)
  - Weighted reranking: Score = α·Recency + β·SemanticSim + γ·SourceAuthority
  - Token-budget greedy selection
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import numpy as np

import chromadb
from chromadb.utils import embedding_functions
from rank_bm25 import BM25Okapi

from data_simulator import WorkspaceDocument
from personalization import UserProfile
from memory import SessionMemory

# ---------------------------------------------------------------------------
# Scoring constants (defaults — overridden per-query by UserProfile)
# ---------------------------------------------------------------------------
ALPHA = 0.30
BETA  = 0.50
GAMMA = 0.20

RECENCY_HALF_LIFE_DAYS = 30

SOURCE_AUTHORITY: dict[str, float] = {
    "docs":     1.0,
    "gmail":    0.8,
    "calendar": 0.7,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _recency_score(ts: datetime, now: Optional[datetime] = None) -> float:
    """Exponential decay: score=1 today, score=0.5 at RECENCY_HALF_LIFE_DAYS."""
    if now is None:
        now = datetime.now(timezone.utc)
    ts_utc = ts.replace(tzinfo=timezone.utc) if ts.tzinfo is None else ts
    days_old = max(0.0, (now - ts_utc).total_seconds() / 86400.0)
    return math.exp(-math.log(2) * days_old / RECENCY_HALF_LIFE_DAYS)


def _count_tokens(text: str) -> int:
    try:
        import tiktoken
        return len(tiktoken.get_encoding("cl100k_base").encode(text))
    except Exception:
        return max(1, int(len(text.split()) * 1.3))


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
@dataclass
class RetrievalCandidate:
    doc: WorkspaceDocument
    score: float
    breakdown: dict = field(default_factory=dict)
    token_count: int = 0
    excluded: bool = False
    exclusion_reason: Optional[str] = None
    memory_note: Optional[str] = None      # set when session memory penalised this doc


@dataclass
class RetrievalResult:
    selected:      list[RetrievalCandidate]
    excluded:      list[RetrievalCandidate]
    all_ranked:    list[RetrievalCandidate]   # all candidates, sorted by score
    expanded_queries: list[str]
    tokens_used:   int
    token_budget:  int


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------
class ContextRouter:
    def __init__(
        self,
        token_budget: int = 2000,
        initial_top_k: int = 10,
        alpha: float = ALPHA,
        beta:  float = BETA,
        gamma: float = GAMMA,
    ):
        self.token_budget   = token_budget
        self.initial_top_k  = initial_top_k
        self.alpha = alpha
        self.beta  = beta
        self.gamma = gamma

        self.documents: list[WorkspaceDocument] = []
        self._corpus_texts: list[str] = []
        self._bm25: Optional[BM25Okapi] = None
        self._collection = None

    # ------------------------------------------------------------------
    # Index
    # ------------------------------------------------------------------
    def index(self, docs: list[WorkspaceDocument]) -> None:
        self.documents     = docs
        self._corpus_texts = [d.full_text() for d in docs]

        # BM25
        self._bm25 = BM25Okapi([t.lower().split() for t in self._corpus_texts])

        # ChromaDB (cosine space, local sentence-transformers model)
        ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        client = chromadb.EphemeralClient()
        self._collection = client.create_collection(
            name="wco_workspace",
            embedding_function=ef,
            metadata={"hnsw:space": "cosine"},
        )
        self._collection.add(
            ids=[d.id for d in docs],
            documents=self._corpus_texts,
            metadatas=[{
                "source": d.source,
                "title":  d.title,
                "ts_unix": d.utc_timestamp().timestamp(),
            } for d in docs],
        )
        print(f"[ContextRouter] Indexed {len(docs)} docs (ChromaDB + BM25).")

    # ------------------------------------------------------------------
    # Query expansion
    # ------------------------------------------------------------------
    def _expand_queries(self, query: str, boost_terms: list[str] | None = None) -> list[str]:
        expanded = [query]
        q = query.lower()

        if any(w in q for w in ("sync", "meeting", "prepare", "prep")):
            expanded += [
                "project status update action items decisions blockers",
                "agenda participants upcoming meeting next steps",
                "recent changes risks open questions",
            ]
        if any(w in q for w in ("project", "proposal", "roadmap", "plan")):
            expanded += [
                "project timeline milestones deliverables budget",
                "latest proposal revision update",
            ]
        if any(w in q for w in ("vendor", "provider", "cloud")):
            expanded += ["vendor evaluation recommendation decision"]

        expanded.append(f"latest recent {query}")
        if boost_terms:
            expanded.append(" ".join(boost_terms))
        return list(dict.fromkeys(expanded))   # preserve order, deduplicate

    # ------------------------------------------------------------------
    # Search backends
    # ------------------------------------------------------------------
    def _vector_search(self, queries: list[str], k: int) -> dict[str, float]:
        k = min(k, len(self.documents))
        scores: dict[str, float] = {}
        for q in queries:
            res = self._collection.query(query_texts=[q], n_results=k)
            for doc_id, dist in zip(res["ids"][0], res["distances"][0]):
                # cosine distance ∈ [0,2] → similarity ∈ [0,1]
                sim = max(0.0, 1.0 - dist / 2.0)
                scores[doc_id] = max(scores.get(doc_id, 0.0), sim)
        return scores

    def _bm25_search(self, queries: list[str], k: int) -> dict[str, float]:
        scores: dict[str, float] = {}
        for q in queries:
            raw = self._bm25.get_scores(q.lower().split())
            top_idx = np.argsort(raw)[::-1][:k]
            max_s = float(raw[top_idx[0]]) if len(top_idx) > 0 and raw[top_idx[0]] > 0 else 0.0
            if max_s <= 0:
                continue
            for idx in top_idx:
                if raw[idx] > 0:
                    doc_id = self.documents[int(idx)].id
                    scores[doc_id] = max(scores.get(doc_id, 0.0), float(raw[idx]) / max_s)
        return scores

    # ------------------------------------------------------------------
    # Retrieve
    # ------------------------------------------------------------------
    def retrieve(
        self,
        query: str,
        user: Optional[UserProfile] = None,
        memory: Optional[SessionMemory] = None,
    ) -> RetrievalResult:
        # Resolve scoring weights from user profile or defaults
        alpha = user.alpha  if user else self.alpha
        beta  = user.beta   if user else self.beta
        gamma = user.gamma  if user else self.gamma
        authority_map = user.source_authority if user else SOURCE_AUTHORITY
        boost_terms   = user.boost_terms      if user else None

        queries = self._expand_queries(query, boost_terms=boost_terms)
        now     = datetime.now(timezone.utc)

        vscore = self._vector_search(queries, k=self.initial_top_k)
        bscore = self._bm25_search(queries,   k=self.initial_top_k)

        doc_map       = {d.id: d for d in self.documents}
        candidate_ids = set(vscore) | set(bscore)

        candidates: list[RetrievalCandidate] = []
        for doc_id in candidate_ids:
            doc = doc_map[doc_id]
            v   = vscore.get(doc_id, 0.0)
            b   = bscore.get(doc_id, 0.0)
            sem = 0.6 * v + 0.4 * b

            rec   = _recency_score(doc.timestamp, now)
            auth  = authority_map.get(doc.source, 0.5)
            score = alpha * rec + beta * sem + gamma * auth

            # Apply session memory penalty for previously seen docs
            mem_note: Optional[str] = None
            if memory:
                score, mem_note = memory.penalize(doc_id, score)

            candidates.append(RetrievalCandidate(
                doc=doc,
                score=score,
                breakdown={
                    "recency":          round(rec,  4),
                    "semantic_sim":     round(sem,  4),
                    "source_authority": auth,
                    "vector_sim":       round(v, 4),
                    "bm25_sim":         round(b, 4),
                    "alpha": alpha, "beta": beta, "gamma": gamma,
                },
                token_count=_count_tokens(doc.full_text()),
                memory_note=mem_note,
            ))

        # Rerank by composite score
        candidates.sort(key=lambda c: c.score, reverse=True)

        # Token-budget greedy selection
        selected: list[RetrievalCandidate] = []
        excluded: list[RetrievalCandidate] = []
        used = 0

        for c in candidates:
            if used + c.token_count <= self.token_budget:
                selected.append(c)
                used += c.token_count
            else:
                reasons: list[str] = []
                reasons.append(
                    f"token budget exceeded ({used + c.token_count} > {self.token_budget})"
                )
                if c.breakdown["recency"] < 0.25:
                    reasons.append(
                        f"stale timestamp (recency={c.breakdown['recency']:.2f})"
                    )
                if c.breakdown["semantic_sim"] < 0.15:
                    reasons.append(
                        f"semantic drift (sim={c.breakdown['semantic_sim']:.2f})"
                    )
                c.excluded = True
                c.exclusion_reason = " + ".join(reasons)
                excluded.append(c)

        return RetrievalResult(
            selected=selected,
            excluded=excluded,
            all_ranked=candidates,
            expanded_queries=queries,
            tokens_used=used,
            token_budget=self.token_budget,
        )
