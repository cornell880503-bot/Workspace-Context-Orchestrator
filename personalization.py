"""
Personalization Layer
Each role gets different source authority weights, scoring weights,
and query boost terms — so the same query returns different ranked
context depending on who is asking.
"""
from __future__ import annotations

from dataclasses import dataclass

ROLE_PROFILES: dict[str, dict] = {
    "pm": {
        "display":          "Product Manager",
        "source_authority": {"docs": 1.0,  "gmail": 0.85, "calendar": 0.90},
        "boost_terms":      ["roadmap", "milestone", "stakeholder", "decision", "OKR", "launch", "timeline"],
        "alpha": 0.25,   # less recency — PMs need historical context too
        "beta":  0.50,
        "gamma": 0.25,   # more authority — formal docs matter more
    },
    "engineer": {
        "display":          "Engineer",
        "source_authority": {"docs": 0.85, "gmail": 1.0,  "calendar": 0.60},
        "boost_terms":      ["API", "spec", "implementation", "bug", "security", "auth", "rate limit"],
        "alpha": 0.35,   # more recency — engineers need the latest specs
        "beta":  0.50,
        "gamma": 0.15,
    },
    "executive": {
        "display":          "Executive",
        "source_authority": {"docs": 1.0,  "gmail": 0.70, "calendar": 0.85},
        "boost_terms":      ["budget", "risk", "strategic", "decision", "summary", "impact", "revenue"],
        "alpha": 0.30,
        "beta":  0.45,
        "gamma": 0.25,
    },
    "designer": {
        "display":          "Designer",
        "source_authority": {"docs": 0.95, "gmail": 0.80, "calendar": 0.75},
        "boost_terms":      ["UX", "mockup", "Figma", "user flow", "feedback", "design"],
        "alpha": 0.30,
        "beta":  0.55,
        "gamma": 0.15,
    },
}

ROLES = list(ROLE_PROFILES.keys())


@dataclass
class UserProfile:
    name: str
    role: str     # "pm" | "engineer" | "executive" | "designer"

    def __post_init__(self) -> None:
        if self.role not in ROLE_PROFILES:
            raise ValueError(f"Unknown role '{self.role}'. Choose from: {ROLES}")
        self._p = ROLE_PROFILES[self.role]

    @property
    def display_role(self) -> str:
        return self._p["display"]

    @property
    def source_authority(self) -> dict[str, float]:
        return self._p["source_authority"]

    @property
    def boost_terms(self) -> list[str]:
        return self._p["boost_terms"]

    @property
    def alpha(self) -> float:
        return self._p["alpha"]

    @property
    def beta(self) -> float:
        return self._p["beta"]

    @property
    def gamma(self) -> float:
        return self._p["gamma"]

    def __str__(self) -> str:
        return f"{self.name} ({self.display_role})"
