#!/usr/bin/env bash
# Runs Codex in auto-edit mode on the current diff.
# Exit 0 = clean (no rewake). Exit 2 = Codex made fixes (rewake Claude).

set -euo pipefail

cd /home/user/Workspace-Context-Orchestrator

DIFF=$(git diff HEAD 2>/dev/null)
if [ -z "$DIFF" ]; then
  DIFF=$(git diff --cached 2>/dev/null)
fi

if [ -z "$DIFF" ]; then
  exit 0
fi

PROMPT="You are a senior code reviewer. Review the following git diff and fix any bugs, security issues, or significant problems directly in the files. Only change what needs fixing — do not refactor working code.

--- DIFF ---
$DIFF
--- END DIFF ---"

OUTPUT=$(codex --approval-mode auto-edit "$PROMPT" 2>&1)

# Check if Codex made any changes
if git diff --quiet 2>/dev/null; then
  exit 0
else
  echo "$OUTPUT"
  exit 2
fi
