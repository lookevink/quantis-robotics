"""Task semantics shared by planner search identity and readiness reporting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from jepa_wm.planner import CandidateTrustRegion
from jepa_wm.planner_readiness import FirstActionThresholds


@dataclass(frozen=True)
class PlannerTaskPolicy:
    proposal_trust_region: CandidateTrustRegion | None = None
    first_action_thresholds: FirstActionThresholds = FirstActionThresholds()

    def to_dict(self) -> dict[str, Any]:
        return {
            "proposal_trust_region": (
                self.proposal_trust_region.to_dict()
                if self.proposal_trust_region is not None
                else None
            ),
            "first_action_thresholds": self.first_action_thresholds.to_dict(),
        }
