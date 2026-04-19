"""
Gemini API Generator
Produces two responses side-by-side:
  - WCO response   : uses WCO-curated, reranked context
  - Naive response : uses only the most-recent docs with no reranking
This contrast is the core demo of why context engineering matters.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

from retriever import RetrievalCandidate
from data_simulator import WorkspaceDocument


SYSTEM_PROMPT = """\
You are a helpful AI assistant integrated into Google Workspace.
You have been given a set of relevant workspace artifacts (emails, documents, calendar events).
Use ONLY the provided context to answer the user's query.
Be concise, specific, and actionable. Format your response in clear bullet points.\
"""


@dataclass
class GenerationResult:
    query:          str
    wco_response:   str
    naive_response: str
    wco_tokens:     int
    naive_tokens:   int


def _build_context_block(candidates: list[RetrievalCandidate]) -> str:
    parts: list[str] = []
    for c in candidates:
        d = c.doc
        ts = d.timestamp.strftime("%Y-%m-%d")
        parts.append(
            f"[{d.source.upper()} | {ts}]\n"
            f"Title: {d.title}\n"
            f"{d.content}"
        )
    return "\n\n---\n\n".join(parts)


def _naive_context(all_docs: list[WorkspaceDocument], token_budget: int) -> list[WorkspaceDocument]:
    """Naive baseline: just sort by recency, no semantic reranking."""
    sorted_docs = sorted(all_docs, key=lambda d: d.timestamp, reverse=True)
    selected: list[WorkspaceDocument] = []
    used = 0
    for d in sorted_docs:
        tokens = int(len(d.full_text().split()) * 1.3)
        if used + tokens <= token_budget:
            selected.append(d)
            used += tokens
    return selected


class GeminiGenerator:
    def __init__(self, model_name: str = "gemini-2.5-flash"):
        api_key = os.environ.get("GEMINI_API_KEY", "")
        if not api_key:
            raise EnvironmentError(
                "GEMINI_API_KEY not set. Run: export GEMINI_API_KEY=your_key"
            )
        try:
            from google import genai
            self._client = genai.Client(api_key=api_key)
            self._model_name = model_name
        except ImportError:
            raise ImportError(
                "google-genai not installed. Run: pip install google-genai"
            )

    def _call(self, query: str, context_block: str) -> str:
        from google import genai as _genai
        prompt = (
            f"{SYSTEM_PROMPT}\n\n"
            f"WORKSPACE CONTEXT:\n\n{context_block}\n\n"
            f"USER QUERY: {query}"
        )
        response = self._client.models.generate_content(
            model=self._model_name,
            contents=prompt,
        )
        return response.text.strip()

    def generate(
        self,
        query: str,
        wco_selected: list[RetrievalCandidate],
        all_docs: list[WorkspaceDocument],
        token_budget: int = 2000,
    ) -> GenerationResult:
        # WCO response
        wco_context  = _build_context_block(wco_selected)
        wco_response = self._call(query, wco_context)

        # Naive response (recency-only, no reranking)
        naive_docs    = _naive_context(all_docs, token_budget)
        naive_context = "\n\n---\n\n".join(
            f"[{d.source.upper()} | {d.timestamp.strftime('%Y-%m-%d')}]\n"
            f"Title: {d.title}\n{d.content}"
            for d in naive_docs
        )
        naive_response = self._call(query, naive_context)

        return GenerationResult(
            query=query,
            wco_response=wco_response,
            naive_response=naive_response,
            wco_tokens=int(len(wco_context.split()) * 1.3),
            naive_tokens=int(len(naive_context.split()) * 1.3),
        )


def print_comparison(result: GenerationResult) -> None:
    W = 72
    print("\n" + "█" * W)
    print("  GEMINI RESPONSE COMPARISON")
    print("  Demonstrates why context quality matters")
    print("█" * W)

    print(f"\n  ┌─ WCO CONTEXT  ({result.wco_tokens} tokens · reranked + personalised)")
    for line in result.wco_response.splitlines():
        print(f"  │  {line}")
    print(f"  └{'─' * (W - 3)}")

    print(f"\n  ┌─ NAIVE CONTEXT  ({result.naive_tokens} tokens · recency-only, no reranking)")
    for line in result.naive_response.splitlines():
        print(f"  │  {line}")
    print(f"  └{'─' * (W - 3)}")

    print("\n  ↑ Same query. Same token budget. Different context quality → different answers.")
    print("█" * W)
