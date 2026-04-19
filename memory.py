"""
Session Memory
Tracks which documents have already been shown this session and
applies a progressive score penalty to surface fresh content on
follow-up queries.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


@dataclass
class SessionMemory:
    # doc_id → how many times shown this session
    _seen: dict[str, int] = field(default_factory=dict)
    _queries: list[tuple[datetime, str]] = field(default_factory=list)

    # 25% score reduction per prior exposure, capped at 3 exposures (75% max)
    PENALTY_PER_VIEW = 0.25
    MAX_VIEWS_COUNTED = 3

    def record_shown(self, doc_ids: list[str]) -> None:
        for doc_id in doc_ids:
            self._seen[doc_id] = self._seen.get(doc_id, 0) + 1

    def record_query(self, query: str) -> None:
        self._queries.append((datetime.now(), query))

    def penalize(self, doc_id: str, score: float) -> tuple[float, Optional[str]]:
        """Return (adjusted_score, reason_string | None)."""
        count = self._seen.get(doc_id, 0)
        if count == 0:
            return score, None
        penalty = self.PENALTY_PER_VIEW * min(count, self.MAX_VIEWS_COUNTED)
        adjusted = round(max(0.0, score * (1.0 - penalty)), 4)
        reason = f"shown {count}× this session → score ×{1-penalty:.2f}"
        return adjusted, reason

    def times_seen(self, doc_id: str) -> int:
        return self._seen.get(doc_id, 0)

    @property
    def stats(self) -> dict:
        return {
            "queries_asked":     len(self._queries),
            "unique_docs_seen":  len(self._seen),
            "total_impressions": sum(self._seen.values()),
        }

    def print_stats(self) -> None:
        s = self.stats
        print(f"  [Memory] {s['queries_asked']} queries · "
              f"{s['unique_docs_seen']} unique docs seen · "
              f"{s['total_impressions']} total impressions this session")
