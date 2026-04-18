# Workspace Context Orchestrator

## Code Review Workflow

Every code change goes through the following cycle before merging:

### Steps

1. **Claude writes code** on a feature branch (e.g. `claude/<feature>`)
2. **Push & open PR** — Claude pushes the branch and creates a PR via GitHub MCP tools
3. **Copilot auto-review** — the `copilot-review.yml` workflow automatically requests GitHub Copilot review the moment a PR is opened
4. **Monitor review** — Claude subscribes to PR activity (`subscribe_pr_activity`) to receive Copilot's comments
5. **Address feedback** — Claude reads each review comment and pushes fixes; re-review happens automatically on new commits
6. **Merge** — only after Copilot approves (or all blocking comments are resolved)

### Claude Code rules

- Never merge a PR before Copilot review is complete
- After creating a PR, always call `subscribe_pr_activity` to watch for review events
- Address every Copilot comment before asking the human for a final decision
- Use `mcp__github__request_copilot_review` to manually re-request review if the automation didn't trigger
