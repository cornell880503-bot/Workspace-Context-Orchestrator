"""
Quality Loss Diagnostics
  - Top-K Misses: docs with high semantic sim that were excluded
  - Reasoning Log: why each excluded doc was dropped
  - Information Density: utilization + source diversity metric
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from retriever import RetrievalCandidate, RetrievalResult


@dataclass
class QualityReport:
    query:              str
    expanded_queries:   list[str]
    total_candidates:   int
    selected_count:     int
    excluded_count:     int
    token_budget:       int
    tokens_used:        int
    information_density: float       # 0–1, higher = denser context
    source_coverage:    dict         # {source: count} in selected set
    top_k_misses:       list[dict]   # high-sim docs that were excluded
    reasoning_log:      list[dict]   # full exclusion log
    score_weights:      dict


class QualityDiagnostics:
    TOP_K_MISS_THRESHOLD = 10   # top-N by semantic sim to watch for misses

    def generate_report(
        self,
        result: RetrievalResult,
        query: str,
        score_weights: dict,
    ) -> QualityReport:

        selected = result.selected
        excluded = result.excluded
        all_ranked = result.all_ranked

        # Source coverage in selected set
        source_coverage: dict[str, int] = {}
        for c in selected:
            source_coverage[c.doc.source] = source_coverage.get(c.doc.source, 0) + 1

        # Information density
        # = token utilization × source diversity bonus
        utilization    = result.tokens_used / result.token_budget if result.token_budget else 0.0
        n_sources      = len(source_coverage)
        max_sources    = 3   # gmail, docs, calendar
        diversity_bonus = n_sources / max_sources
        information_density = round(utilization * (0.7 + 0.3 * diversity_bonus), 4)

        # Top-K misses: docs that ranked in top-K by pure semantic sim
        # but were NOT in the final selected set
        selected_ids = {c.doc.id for c in selected}
        by_semantic = sorted(all_ranked, key=lambda c: c.breakdown["semantic_sim"], reverse=True)
        top_k_pool  = by_semantic[: self.TOP_K_MISS_THRESHOLD]
        top_k_misses = [
            {
                "id":               c.doc.id,
                "source":           c.doc.source,
                "title":            c.doc.title,
                "semantic_sim":     c.breakdown["semantic_sim"],
                "recency":          c.breakdown["recency"],
                "composite_score":  round(c.score, 4),
                "token_count":      c.token_count,
                "exclusion_reason": c.exclusion_reason or "reranked below token-budget cutoff",
            }
            for c in top_k_pool
            if c.doc.id not in selected_ids
        ]

        # Reasoning log for every excluded doc
        reasoning_log = [
            {
                "id":      c.doc.id,
                "source":  c.doc.source,
                "title":   c.doc.title,
                "score":   round(c.score, 4),
                "reason":  c.exclusion_reason or "below selection threshold",
                "recency": c.breakdown["recency"],
                "semantic_sim": c.breakdown["semantic_sim"],
                "source_authority": c.breakdown["source_authority"],
            }
            for c in excluded
        ]

        return QualityReport(
            query=query,
            expanded_queries=result.expanded_queries,
            total_candidates=len(all_ranked),
            selected_count=len(selected),
            excluded_count=len(excluded),
            token_budget=result.token_budget,
            tokens_used=result.tokens_used,
            information_density=information_density,
            source_coverage=source_coverage,
            top_k_misses=top_k_misses,
            reasoning_log=reasoning_log,
            score_weights=score_weights,
        )

    def to_dataframe(self, report: QualityReport) -> dict[str, pd.DataFrame]:
        """Export report tables as Pandas DataFrames for downstream analysis."""
        misses_df = pd.DataFrame(report.top_k_misses) if report.top_k_misses else pd.DataFrame()
        log_df    = pd.DataFrame(report.reasoning_log) if report.reasoning_log else pd.DataFrame()
        return {"top_k_misses": misses_df, "reasoning_log": log_df}

    # ------------------------------------------------------------------
    # Pretty-print
    # ------------------------------------------------------------------
    def print_report(self, report: QualityReport) -> None:
        W = 72

        def _bar(label: str, value: float, width: int = 30) -> str:
            filled = int(value * width)
            return f"{label} [{'█' * filled}{'░' * (width - filled)}] {value:.1%}"

        print("\n" + "═" * W)
        print("  QUALITY LOSS DIAGNOSTICS REPORT")
        print("═" * W)
        print(f"  Query          : {report.query}")
        print(f"  Sub-queries    : {len(report.expanded_queries)} generated")
        for q in report.expanded_queries[1:]:
            print(f"                   └─ {q}")
        print(f"\n  Candidates     : {report.total_candidates}")
        print(f"  Selected       : {report.selected_count}  ({report.tokens_used} / {report.token_budget} tokens)")
        print(f"  Excluded       : {report.excluded_count}")
        print(f"  Source coverage: {report.source_coverage}")
        print(f"\n  {_bar('Token utilization ', report.tokens_used / report.token_budget)}")
        print(f"  {_bar('Information density', report.information_density)}")

        w = report.score_weights
        print(f"\n  Scoring weights: α={w['alpha']} (recency)  "
              f"β={w['beta']} (semantic)  γ={w['gamma']} (authority)")

        # --- Top-K Misses ---
        if report.top_k_misses:
            print(f"\n  ┌─ TOP-K MISSES ({len(report.top_k_misses)} docs — high semantic sim but excluded)")
            for m in report.top_k_misses:
                src   = m["source"].upper().ljust(8)
                title = m["title"][:48].ljust(48)
                print(f"  │  [{src}] {title}")
                print(f"  │          sem={m['semantic_sim']:.3f}  "
                      f"recency={m['recency']:.3f}  "
                      f"composite={m['composite_score']:.3f}  "
                      f"tokens={m['token_count']}")
                print(f"  │          ↳ WHY EXCLUDED: {m['exclusion_reason']}")
            print("  └" + "─" * (W - 3))
        else:
            print("\n  ✓ No top-K misses — all high-relevance docs fit within budget.")

        # --- Reasoning Log ---
        if report.reasoning_log:
            print(f"\n  ┌─ FULL EXCLUSION LOG")
            for e in report.reasoning_log:
                src   = e["source"].upper().ljust(8)
                title = e["title"][:48].ljust(48)
                print(f"  │  [{src}] {title}")
                print(f"  │          score={e['score']:.3f}  ↳ {e['reason']}")
            print("  └" + "─" * (W - 3))

        print("═" * W)
