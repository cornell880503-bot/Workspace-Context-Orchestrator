# Workspace Context Orchestrator (WCO)

A prototype demonstrating **Dynamic Context Engineering** and **Quality Loss Diagnostics** for AI assistants operating across multi-source workspace data (Gmail, Google Docs, Calendar).

Built as a technical proof-of-concept for the Google Workspace Context PM role.

---

## The Problem

When a user asks an AI assistant *"Prepare me for my project sync"*, the assistant needs to pull from hundreds of emails, documents, and calendar events — but its context window only fits a fraction of them.

Simple keyword search fails here: a 6-month-old sync checklist is *semantically identical* to today's sync prep email, but including the old one wastes precious tokens and injects stale context.

**WCO solves this by:**
1. Ranking every artifact using a composite score (recency × relevance × source authority)
2. Fitting the highest-value artifacts within a token budget
3. Explaining exactly which artifacts were excluded and why

---

## Architecture

```
User Query
    │
    ▼
Multi-Query Expansion ──── personalised boost terms (role-aware)
    │
    ├─── BM25 Keyword Search ──┐
    │                           ├── Hybrid Score (60% vector + 40% BM25)
    └─── ChromaDB Vector Search ┘
                │
                ▼
        Weighted Reranker
        Score = α·Recency + β·SemanticSim + γ·SourceAuthority
        (weights differ per user role)
                │
                ▼
    Session Memory Penalty  ← penalises docs shown in prior queries
                │
                ▼
    Token Budget Selector  ← greedy selection within 2000-token limit
                │
        ┌───────┴────────┐
        ▼                ▼
  Selected Context   Excluded Artifacts
                          │
                          ▼
              Quality Loss Diagnostics
              • Top-K Misses
              • Reasoning Log
              • Information Density
```

---

## Features

### Dynamic Context Engineering
- **Hybrid search**: BM25 (keyword) + ChromaDB (semantic embeddings via `all-MiniLM-L6-v2`)
- **Multi-query expansion**: one query becomes 5–8 sub-queries covering different aspects
- **Weighted reranking**:

$$Score = \alpha \cdot Recency + \beta \cdot SemanticSim + \gamma \cdot SourceAuthority$$

  Default weights: α=0.30, β=0.50, γ=0.20

- **Token budget enforcement**: greedy selection capped at 2000 tokens (configurable)

### Personalization Layer
Each user role gets different scoring weights and source authority — the same query returns a different ranked context depending on who is asking:

| Role | α (Recency) | β (Semantic) | γ (Authority) | Docs | Gmail | Calendar |
|------|------------|--------------|---------------|------|-------|----------|
| PM | 0.25 | 0.50 | 0.25 | 1.0 | 0.85 | 0.90 |
| Engineer | 0.35 | 0.50 | 0.15 | 0.85 | 1.0 | 0.60 |
| Executive | 0.30 | 0.45 | 0.25 | 1.0 | 0.70 | 0.85 |
| Designer | 0.30 | 0.55 | 0.15 | 0.95 | 0.80 | 0.75 |

### Session Memory
Tracks which documents have already been shown this session. Each re-exposure reduces the document's score by 25% (capped at 3×), surfacing fresh content on follow-up queries instead of repeating the same artifacts.

### Quality Loss Diagnostics
- **Top-K Misses**: documents with high semantic similarity that didn't make the final cut
- **Reasoning Log**: plain-English explanation for every exclusion (token limit / stale timestamp / semantic drift)
- **Information Density**: composite metric of token utilisation × source diversity
- **CSV export** of misses for downstream analysis

---

## Quickstart

### Prerequisites
- Python 3.10+
- ~80 MB disk space (embedding model, downloaded on first run)

### Setup

```bash
git clone https://github.com/cornell880503-bot/Workspace-Context-Orchestrator.git
cd Workspace-Context-Orchestrator
git checkout claude/codex-review-integration-38slg

python3 -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Run

```bash
# Default query, no personalization
python3 main.py

# With role personalization
python3 main.py --role pm --name Sarah --query "Prepare for my project sync"
python3 main.py --role engineer --name John --query "What are the security risks?"
python3 main.py --role executive --query "What decisions need my sign-off?"

# Smaller token budget to trigger exclusion diagnostics
python3 main.py --role pm --budget 400 --query "Prepare for my project sync"

# Interactive REPL (memory persists across queries)
python3 main.py --role pm --interactive

# Gemini comparison: WCO-curated context vs naive recency-only context
export GEMINI_API_KEY=your_key   # or add to ~/.zshrc
python3 main.py --role pm --generate --query "Prepare for my project sync"
```

### CLI flags

| Flag | Short | Default | Description |
|------|-------|---------|-------------|
| `--query` | `-q` | `"Prepare for my project sync"` | Natural-language query |
| `--role` | `-r` | none | `pm` / `engineer` / `executive` / `designer` |
| `--name` | `-n` | role name | Display name for the user |
| `--budget` | `-b` | `2000` | Token budget for context window |
| `--interactive` | `-i` | off | REPL mode with persistent session memory |
| `--generate` | `-g` | off | Call Gemini and show WCO vs Naive response comparison |

---

## Project Structure

```
├── mock_data.json        # 16 simulated Gmail / Docs / Calendar artifacts
├── data_simulator.py     # WorkspaceDocument dataclass + JSON loader
├── retriever.py          # Core pipeline: hybrid search → rerank → token budget
├── personalization.py    # UserProfile with role-based scoring weights
├── memory.py             # SessionMemory: cross-query seen-doc penalty
├── diagnostics.py        # Quality Loss report: Top-K Misses, Reasoning Log
├── main.py               # CLI entry point
└── requirements.txt
```

---

## Sample Output

```
════════════════════════════════════════════════════════════════════════
  SELECTED CONTEXT  ·  'Prepare for my project sync'  ·  Sarah (Product Manager)
════════════════════════════════════════════════════════════════════════
  Weights: α=0.25 (recency)  β=0.50 (semantic)  γ=0.25 (authority)

  [1] [GMAIL]  Re: Project Alpha Sync — This Thursday 2pm
       score=0.871  recency=0.950  semantic=0.860  authority=0.85  tokens=62
  [2] [GMAIL]  URGENT: Security Audit Results — Review Before Thursday Sync
       score=0.822  recency=0.934  semantic=0.774  authority=0.85  tokens=79
  ...

════════════════════════════════════════════════════════════════════════
  QUALITY LOSS DIAGNOSTICS REPORT
════════════════════════════════════════════════════════════════════════
  ┌─ TOP-K MISSES (high semantic sim → excluded)
  │  [GMAIL  ] Project Sync Prep — Checklist Template
  │          sem=0.896  recency=0.032  composite=0.617
  │          ↳ WHY EXCLUDED: stale timestamp (recency=0.03)
  └───────────────────────────────────────────────────────────────────
```

---

## Technical Notes

**Recency decay**: exponential with 30-day half-life
$$Recency(t) = e^{-\ln 2 \cdot \frac{days\_old}{30}}$$

**Semantic similarity**: cosine distance from ChromaDB (60%) + BM25 normalised score (40%)

**Token counting**: uses `tiktoken` (`cl100k_base`) if available, otherwise word-count approximation (×1.3)

**Memory penalty**: score × (1 − 0.25 × min(views, 3)), so a doc shown 3× has its score halved
