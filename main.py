#!/usr/bin/env python3
"""
Workspace Context Orchestrator — CLI
Usage:
    python main.py
    python main.py --query "Prepare for my project sync" --role pm --name Sarah
    python main.py --role engineer --generate
    python main.py --budget 800 --role engineer --interactive
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from data_simulator import WorkspaceDataSimulator
from retriever import ContextRouter
from diagnostics import QualityDiagnostics
from personalization import UserProfile, ROLES
from memory import SessionMemory


DEFAULT_QUERY  = "Prepare for my project sync"
DEFAULT_BUDGET = 2000
DATA_PATH      = Path(__file__).parent / "mock_data.json"


def _banner() -> None:
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║        Workspace Context Orchestrator  (WCO)  —  Prototype v1       ║
║   Dynamic Context Engineering  ·  Personalization  ·  Memory        ║
╚══════════════════════════════════════════════════════════════════════╝""")


def _ask_user_profile() -> UserProfile:
    print(f"  Roles: {', '.join(ROLES)}")
    name = input("  Your name  > ").strip() or "User"
    while True:
        role = input(f"  Your role  > ").strip().lower()
        if role in ROLES:
            break
        print(f"  Please choose from: {ROLES}")
    return UserProfile(name=name, role=role)


def _print_selected(result, query: str, user: UserProfile | None) -> None:
    W = 72
    who = f" · {user}" if user else ""
    print(f"\n{'═' * W}")
    print(f"  SELECTED CONTEXT  ·  '{query}'{who}")
    print(f"{'═' * W}")

    if not result.selected:
        print("  (no documents selected)")
        print("═" * W)
        return

    bd = result.selected[0].breakdown
    print(f"  Weights: α={bd['alpha']} (recency)  β={bd['beta']} (semantic)  γ={bd['gamma']} (authority)\n")

    for i, c in enumerate(result.selected, 1):
        src     = c.doc.source.upper()
        ts      = c.doc.timestamp.strftime("%Y-%m-%d")
        snippet = c.doc.content[:200].replace("\n", " ")
        if len(c.doc.content) > 200:
            snippet += "…"

        mem_tag = f"  ⟳ {c.memory_note}" if c.memory_note else ""
        print(f"  [{i}] [{src}]  {c.doc.title}")
        print(f"       score={c.score:.3f}  recency={c.breakdown['recency']:.3f}  "
              f"semantic={c.breakdown['semantic_sim']:.3f}  "
              f"authority={c.breakdown['source_authority']}  "
              f"tokens={c.token_count}  date={ts}{mem_tag}")
        print(f"       {snippet}")

    print(f"\n  Tokens used : {result.tokens_used} / {result.token_budget}")
    print("═" * W)


def _print_excluded_summary(result) -> None:
    if not result.excluded:
        return
    print("\n  EXCLUDED ARTIFACTS")
    print("  " + "─" * 68)
    for c in result.excluded:
        src   = c.doc.source.upper().ljust(8)
        title = c.doc.title[:52].ljust(52)
        print(f"  [{src}] {title}  score={c.score:.3f}")
        print(f"            ↳ {c.exclusion_reason}")


def run(
    query: str,
    token_budget: int,
    router: ContextRouter,
    user: UserProfile | None,
    memory: SessionMemory,
    generate: bool = False,
) -> None:
    result = router.retrieve(query, user=user, memory=memory)

    _print_selected(result, query, user)
    _print_excluded_summary(result)

    diag   = QualityDiagnostics()
    report = diag.generate_report(
        result=result,
        query=query,
        score_weights={
            "alpha": user.alpha  if user else router.alpha,
            "beta":  user.beta   if user else router.beta,
            "gamma": user.gamma  if user else router.gamma,
        },
    )
    diag.print_report(report)

    # Update memory with what was shown
    memory.record_query(query)
    memory.record_shown([c.doc.id for c in result.selected])
    memory.print_stats()

    # Export CSV if misses exist
    dfs = diag.to_dataframe(report)
    if not dfs["top_k_misses"].empty:
        out = Path("wco_misses.csv")
        dfs["top_k_misses"].to_csv(out, index=False)
        print(f"\n  [CSV] Top-K misses → {out}")

    # Gemini comparison
    if generate:
        import os as _os
        _key = _os.environ.get("GEMINI_API_KEY", "")
        print(f"\n  [Gemini DEBUG] GEMINI_API_KEY present: {bool(_key)} "
              f"(len={len(_key)})")
        try:
            import google.genai as _probe
            print(f"  [Gemini DEBUG] google.genai loaded from: {_probe.__file__}")
        except ImportError as _e:
            print(f"  [Gemini DEBUG] google.genai import failed: {_e}")

        from generator import GeminiGenerator, print_comparison
        try:
            print("\n  [Gemini] Generating responses …")
            gen = GeminiGenerator()
            comparison = gen.generate(
                query=query,
                wco_selected=result.selected,
                all_docs=router.documents,
                token_budget=token_budget,
            )
            print_comparison(comparison)
        except (EnvironmentError, ImportError) as e:
            print(f"\n  [Gemini] Skipped: {e}")


def main() -> None:
    _banner()

    parser = argparse.ArgumentParser(description="Workspace Context Orchestrator")
    parser.add_argument("--query",  "-q", type=str,  default="")
    parser.add_argument("--budget", "-b", type=int,  default=DEFAULT_BUDGET)
    parser.add_argument("--role",   "-r", type=str,  default="", choices=ROLES + [""])
    parser.add_argument("--name",   "-n", type=str,  default="")
    parser.add_argument("--interactive", "-i", action="store_true")
    parser.add_argument("--generate",    "-g", action="store_true", help="Call Gemini API and show WCO vs Naive comparison")
    args = parser.parse_args()

    if not DATA_PATH.exists():
        print(f"[ERROR] {DATA_PATH} not found.")
        sys.exit(1)

    # Load & index (once, shared across all queries)
    print(f"\n[1/3] Loading workspace data …")
    sim  = WorkspaceDataSimulator(str(DATA_PATH))
    docs = sim.load()
    sim.summary(docs)

    print("[2/3] Indexing …  (first run downloads ~80 MB model)")
    router = ContextRouter(token_budget=args.budget)
    router.index(docs)

    # Build user profile
    user: UserProfile | None = None
    if args.role:
        name = args.name or args.role.capitalize()
        user = UserProfile(name=name, role=args.role)
        print(f"\n  User: {user}")
    elif args.interactive:
        print("\n[Personalization]")
        user = _ask_user_profile()
        print(f"  Welcome, {user}!\n")

    memory = SessionMemory()

    print("[3/3] Ready.\n")

    if args.interactive:
        print(f"  Token budget: {args.budget}  |  Type 'quit' to exit\n")
        while True:
            try:
                raw = input("Query > ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nBye.")
                break
            if raw.lower() in ("quit", "exit", "q"):
                print("Bye.")
                break
            query = raw or DEFAULT_QUERY
            if not raw:
                print(f"  [default: '{query}']")
            run(query, args.budget, router, user, memory, generate=args.generate)
    else:
        query = args.query or DEFAULT_QUERY
        if not args.query:
            print(f"  No --query provided, using default: '{query}'\n")
        run(query, args.budget, router, user, memory, generate=args.generate)


if __name__ == "__main__":
    main()
