#!/usr/bin/env python3
"""
Workspace Context Orchestrator — CLI
Usage:
    python main.py
    python main.py --query "Prepare for my project sync"
    python main.py --budget 1000 --query "What are the vendor risks?"
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from data_simulator import WorkspaceDataSimulator
from retriever import ContextRouter, ALPHA, BETA, GAMMA
from diagnostics import QualityDiagnostics


DEFAULT_QUERY  = "Prepare for my project sync"
DEFAULT_BUDGET = 2000
DATA_PATH      = Path(__file__).parent / "mock_data.json"


def _banner() -> None:
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║        Workspace Context Orchestrator  (WCO)  —  Prototype v1       ║
║   Dynamic Context Engineering  ·  Quality Loss Diagnostics          ║
╚══════════════════════════════════════════════════════════════════════╝""")


def _print_selected(result, query: str) -> None:
    W = 72
    print("\n" + "═" * W)
    print(f"  SELECTED CONTEXT  ·  query: '{query}'")
    print("═" * W)

    if not result.selected:
        print("  (no documents selected)")
        print("═" * W)
        return

    for i, c in enumerate(result.selected, 1):
        src   = c.doc.source.upper()
        title = c.doc.title
        ts    = c.doc.timestamp.strftime("%Y-%m-%d")
        score = c.score
        toks  = c.token_count
        snippet = c.doc.content[:220].replace("\n", " ")
        if len(c.doc.content) > 220:
            snippet += "…"

        print(f"\n  [{i}] [{src}]  {title}")
        print(f"       score={score:.3f}  recency={c.breakdown['recency']:.3f}  "
              f"semantic={c.breakdown['semantic_sim']:.3f}  "
              f"authority={c.breakdown['source_authority']}  tokens={toks}  date={ts}")
        print(f"       {snippet}")

    print(f"\n  Tokens used : {result.tokens_used} / {result.token_budget}")
    print("═" * W)


def _print_excluded_summary(result) -> None:
    if not result.excluded:
        return
    print("\n  EXCLUDED ARTIFACTS (not in final context window)")
    print("  " + "─" * 68)
    for c in result.excluded:
        src   = c.doc.source.upper().ljust(8)
        title = c.doc.title[:52].ljust(52)
        print(f"  [{src}] {title}  score={c.score:.3f}")
        print(f"            ↳ {c.exclusion_reason}")


def run(query: str, token_budget: int) -> None:
    if not DATA_PATH.exists():
        print(f"[ERROR] Data file not found: {DATA_PATH}")
        sys.exit(1)

    # --- Load & index ---
    print(f"\n[1/3] Loading workspace data from {DATA_PATH.name} …")
    sim  = WorkspaceDataSimulator(str(DATA_PATH))
    docs = sim.load()
    sim.summary(docs)

    print("[2/3] Indexing (ChromaDB embeddings + BM25) …  (first run downloads ~80 MB model)")
    router = ContextRouter(token_budget=token_budget)
    router.index(docs)

    # --- Retrieve ---
    print(f"[3/3] Retrieving context for: '{query}'\n")
    result = router.retrieve(query)

    # --- Display ---
    _print_selected(result, query)
    _print_excluded_summary(result)

    diag   = QualityDiagnostics()
    report = diag.generate_report(
        result=result,
        query=query,
        score_weights={"alpha": router.alpha, "beta": router.beta, "gamma": router.gamma},
    )
    diag.print_report(report)

    # Optional: export CSV for analysis
    dfs = diag.to_dataframe(report)
    if not dfs["top_k_misses"].empty:
        out = Path("wco_misses.csv")
        dfs["top_k_misses"].to_csv(out, index=False)
        print(f"\n  [CSV] Top-K misses exported → {out}")


def main() -> None:
    _banner()

    parser = argparse.ArgumentParser(description="Workspace Context Orchestrator")
    parser.add_argument("--query",  "-q", type=str, default="",   help="Natural-language query")
    parser.add_argument("--budget", "-b", type=int, default=DEFAULT_BUDGET, help="Token budget (default 2000)")
    parser.add_argument("--interactive", "-i", action="store_true", help="Interactive REPL mode")
    args = parser.parse_args()

    if args.interactive:
        # Interactive REPL
        print(f"  Token budget: {args.budget}   |   Type 'quit' to exit\n")
        while True:
            try:
                raw = input("Query > ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nBye.")
                break
            if raw.lower() in ("quit", "exit", "q", ""):
                if raw == "":
                    raw = DEFAULT_QUERY
                    print(f"  [Using default: '{raw}']")
                else:
                    print("Bye.")
                    break
            run(raw, args.budget)
    else:
        query = args.query or DEFAULT_QUERY
        if not args.query:
            print(f"  No --query provided, using default: '{query}'\n")
        run(query, args.budget)


if __name__ == "__main__":
    main()
