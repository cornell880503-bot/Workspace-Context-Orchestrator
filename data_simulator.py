from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path


@dataclass
class WorkspaceDocument:
    id: str
    source: str          # "gmail" | "docs" | "calendar"
    title: str
    content: str
    timestamp: datetime
    metadata: dict = field(default_factory=dict)

    def utc_timestamp(self) -> datetime:
        if self.timestamp.tzinfo is None:
            return self.timestamp.replace(tzinfo=timezone.utc)
        return self.timestamp

    def full_text(self) -> str:
        return f"{self.title}\n{self.content}"


class WorkspaceDataSimulator:
    def __init__(self, data_path: str = "mock_data.json"):
        self.data_path = Path(data_path)

    def load(self) -> list[WorkspaceDocument]:
        with open(self.data_path, encoding="utf-8") as f:
            raw: list[dict] = json.load(f)

        docs: list[WorkspaceDocument] = []
        for item in raw:
            docs.append(WorkspaceDocument(
                id=item["id"],
                source=item["source"],
                title=item["title"],
                content=item["content"],
                timestamp=datetime.fromisoformat(item["timestamp"]),
                metadata=item.get("metadata", {}),
            ))

        return docs

    def summary(self, docs: list[WorkspaceDocument]) -> None:
        by_source: dict[str, int] = {}
        for d in docs:
            by_source[d.source] = by_source.get(d.source, 0) + 1
        print(f"[DataSimulator] Loaded {len(docs)} documents: {by_source}")
