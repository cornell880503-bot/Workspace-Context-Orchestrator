#!/usr/bin/env bash
# Runs `codex review` on the current changes.
# Exit 0 = no issues. Exit 2 = Codex flagged problems (rewake Claude).

set -euo pipefail

cd /home/user/Workspace-Context-Orchestrator

# Determine what to review: uncommitted changes, then last commit
HAS_UNCOMMITTED=$(git status --porcelain 2>/dev/null)

if [ -n "$HAS_UNCOMMITTED" ]; then
  OUTPUT=$(codex review --uncommitted 2>&1)
else
  LAST_COMMIT=$(git rev-parse HEAD 2>/dev/null)
  if [ -z "$LAST_COMMIT" ]; then
    exit 0
  fi
  OUTPUT=$(codex review --commit "$LAST_COMMIT" 2>&1)
fi

echo "$OUTPUT"

# Rewake Claude if Codex flagged any issues (non-empty output with "issue"/"problem"/"error"/"fix")
if echo "$OUTPUT" | grep -qiE "(issue|problem|bug|error|vulnerabilit|should|recommend|fix|concern)"; then
  exit 2
fi

exit 0
