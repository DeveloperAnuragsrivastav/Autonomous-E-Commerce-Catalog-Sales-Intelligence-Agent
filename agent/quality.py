"""Data-quality log: every anomaly the pipeline detects, and what it did about it.

Entries are aggregated (one entry per issue type, with a count and examples) so the report
stays readable on 10k-row inputs.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class Issue:
    sheet: str
    field: str
    issue: str
    action: str
    count: int = 0
    examples: list[str] = field(default_factory=list)
    severity: str = "info"  # info | warning | error


class QualityLog:
    MAX_EXAMPLES = 5

    def __init__(self) -> None:
        self._issues: dict[tuple[str, str, str], Issue] = {}

    def add(
        self,
        sheet: str,
        field: str,
        issue: str,
        action: str,
        *,
        count: int = 1,
        example: str | None = None,
        examples: list[str] | None = None,
        severity: str = "info",
    ) -> None:
        if count <= 0:
            return
        key = (sheet, field, issue)
        entry = self._issues.setdefault(key, Issue(sheet, field, issue, action, severity=severity))
        entry.count += count
        for ex in ([example] if example else []) + (examples or []):
            if len(entry.examples) < self.MAX_EXAMPLES and ex not in entry.examples:
                entry.examples.append(ex)

    @property
    def issues(self) -> list[Issue]:
        order = {"error": 0, "warning": 1, "info": 2}
        return sorted(self._issues.values(), key=lambda i: (i.sheet, order[i.severity], -i.count))

    def to_records(self) -> list[dict]:
        return [{**asdict(i), "examples": "; ".join(i.examples)} for i in self.issues]
