"""Typed no-actuation safety evidence for one shadow candidate."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any

from jepa_wm.action import DroidAction, DroidActionScale
from jepa_wm.control_safety import ACTION_SCALES, SafetyProjectionAttempt
from jepa_wm.shadow_planning import CandidateAuthority


SHADOW_SAFETY_SCHEMA = "quantis.jepa_wm_shadow_safety.v1"


@dataclass(frozen=True)
class ShadowSafetyEvidence:
    observation_id: int
    evaluated_at_unix_seconds: float
    counterfactual_as_of_unix_seconds: float
    planned_actions: tuple[DroidAction, ...]
    attempts: tuple[SafetyProjectionAttempt, ...]
    selected_action_scale: DroidActionScale | None
    authority: CandidateAuthority = CandidateAuthority.SHADOW_ONLY

    def __post_init__(self) -> None:
        if (
            isinstance(self.observation_id, bool)
            or self.observation_id <= 0
            or not isfinite(self.evaluated_at_unix_seconds)
            or not isfinite(self.counterfactual_as_of_unix_seconds)
            or self.evaluated_at_unix_seconds
            < self.counterfactual_as_of_unix_seconds
            or len(self.planned_actions) != 3
            or not self.attempts
            or self.authority is not CandidateAuthority.SHADOW_ONLY
        ):
            raise ValueError("shadow safety evidence is invalid")
        selected = tuple(
            attempt
            for attempt in self.attempts
            if attempt.scale == self.selected_action_scale and attempt.gate.passed
        )
        if (self.selected_action_scale is None) == bool(selected):
            raise ValueError("shadow safety selection is inconsistent")
        if tuple(attempt.scale for attempt in self.attempts) != ACTION_SCALES[
            : len(self.attempts)
        ]:
            raise ValueError("shadow safety projection order is invalid")
        if any(
            attempt.gate.observation_id != self.observation_id
            for attempt in self.attempts
        ):
            raise ValueError("shadow safety observation binding is invalid")
        if any(attempt.gate.passed for attempt in self.attempts[:-1]):
            raise ValueError("shadow safety continued after a passing projection")
        if self.attempts[-1].gate.passed:
            if self.selected_action_scale != self.attempts[-1].scale:
                raise ValueError("shadow safety selected the wrong projection")
        elif self.selected_action_scale is not None or len(self.attempts) != len(
            ACTION_SCALES
        ):
            raise ValueError("shadow safety stopped before exhausting projections")

    @property
    def passed(self) -> bool:
        return self.selected_action_scale is not None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": SHADOW_SAFETY_SCHEMA,
            "observation_id": self.observation_id,
            "evaluated_at_unix_seconds": self.evaluated_at_unix_seconds,
            "counterfactual_as_of_unix_seconds": (
                self.counterfactual_as_of_unix_seconds
            ),
            "planned_actions": [list(action.values) for action in self.planned_actions],
            "attempts": [attempt.to_dict() for attempt in self.attempts],
            "selected_action_scale": (
                self.selected_action_scale.to_dict()
                if self.selected_action_scale is not None
                else None
            ),
            "passed": self.passed,
            "authority": self.authority.value,
        }

    @classmethod
    def from_dict(cls, payload: Any) -> ShadowSafetyEvidence:
        if (
            not isinstance(payload, dict)
            or payload.get("schema") != SHADOW_SAFETY_SCHEMA
        ):
            raise ValueError("shadow safety evidence schema is invalid")
        try:
            authority = CandidateAuthority(payload["authority"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("shadow safety authority is invalid") from error
        try:
            selected = payload["selected_action_scale"]
            # Legacy v1 artifacts used the counterfactual timestamp as the
            # evaluation timestamp. Accept them without propagating that
            # ambiguity into newly written evidence.
            counterfactual_as_of = float(
                payload.get(
                    "counterfactual_as_of_unix_seconds",
                    payload["evaluated_at_unix_seconds"],
                )
            )
            evidence = cls(
                observation_id=int(payload["observation_id"]),
                evaluated_at_unix_seconds=float(payload["evaluated_at_unix_seconds"]),
                counterfactual_as_of_unix_seconds=counterfactual_as_of,
                planned_actions=tuple(
                    DroidAction(tuple(values)) for values in payload["planned_actions"]
                ),
                attempts=tuple(
                    SafetyProjectionAttempt.from_dict(attempt)
                    for attempt in payload["attempts"]
                ),
                selected_action_scale=(
                    DroidActionScale.from_payload(selected)
                    if selected is not None
                    else None
                ),
                authority=authority,
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("shadow safety evidence is incomplete") from error
        if payload.get("passed") is not evidence.passed:
            raise ValueError("shadow safety pass result is inconsistent")
        return evidence
