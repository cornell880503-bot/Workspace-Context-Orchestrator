#!/usr/bin/env bash
# Run OpenAI Codex to review uncommitted or staged changes.
# Usage: ./scripts/codex-review.sh [base-ref]
#   base-ref defaults to HEAD (reviews unstaged+staged changes)
#   Pass a commit SHA or branch to diff against that ref instead.

set -euo pipefail

BASE="${1:-HEAD}"

DIFF=$(git diff "$BASE" 2>/dev/null)
if [ -z "$DIFF" ]; then
  DIFF=$(git diff --cached 2>/dev/null)
fi

if [ -z "$DIFF" ]; then
  echo "No changes to review."
  exit 0
fi

PROMPT="You are a senior code reviewer. Review the following git diff carefully.
Identify: bugs, security issues, performance problems, style issues, and improvements.
Be concise and actionable. Format as a numbered list.

--- DIFF ---
$DIFF
--- END DIFF ---"

echo "Running Codex review..."
echo "---"
codex --approval-mode suggest-only "$PROMPT"
