"""Bounded lineage position for an autonomous insertion rollout."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Any, Mapping

from jepa_wm.control_policy import ControlExecutionPolicy
from jepa_wm.identifiers import validate_safe_identifier


@dataclass(frozen=True)
class InsertionRolloutProfile:
    """One named, immutable rollout cap exposed to orchestration."""

    name: str
    maximum_steps: int

    def __post_init__(self) -> None:
        validate_safe_identifier(self.name)
        if (
            isinstance(self.maximum_steps, bool)
            or not isinstance(self.maximum_steps, int)
            or self.maximum_steps < 1
        ):
            raise ValueError("insertion rollout profile is invalid")


TWO_STEP_INSERTION_ROLLOUT = InsertionRolloutProfile("two-step", 2)
DEMO_INSERTION_ROLLOUT = InsertionRolloutProfile("demo", 4)
APPROACH_INSERTION_ROLLOUT = InsertionRolloutProfile("approach", 8)
ALIGNMENT_INSERTION_ROLLOUT = InsertionRolloutProfile("alignment", 12)
PRE_INSERTION_ROLLOUT = InsertionRolloutProfile("pre-insertion", 32)
CONTACT_INSERTION_ROLLOUT = InsertionRolloutProfile("contact-insertion", 96)
GRASP_TO_INSERTION_ROLLOUT = InsertionRolloutProfile("grasp-to-insertion", 168)
INSERTION_ROLLOUT_PROFILES = (
    TWO_STEP_INSERTION_ROLLOUT,
    DEMO_INSERTION_ROLLOUT,
    APPROACH_INSERTION_ROLLOUT,
    ALIGNMENT_INSERTION_ROLLOUT,
    PRE_INSERTION_ROLLOUT,
    CONTACT_INSERTION_ROLLOUT,
    GRASP_TO_INSERTION_ROLLOUT,
)
INSERTION_ROLLOUT_EXTENSIONS = (
    (DEMO_INSERTION_ROLLOUT, APPROACH_INSERTION_ROLLOUT),
    (APPROACH_INSERTION_ROLLOUT, ALIGNMENT_INSERTION_ROLLOUT),
    (ALIGNMENT_INSERTION_ROLLOUT, PRE_INSERTION_ROLLOUT),
    (PRE_INSERTION_ROLLOUT, CONTACT_INSERTION_ROLLOUT),
)


def insertion_rollout_profile(name: str) -> InsertionRolloutProfile:
    for profile in INSERTION_ROLLOUT_PROFILES:
        if profile.name == name:
            return profile
    raise ValueError("insertion rollout profile is unsupported")


def is_insertion_rollout_policy(policy: ControlExecutionPolicy) -> bool:
    return policy in (
        ControlExecutionPolicy.INSERTION_SAFETY_EVALUATION,
        ControlExecutionPolicy.INSERTION_RESET_TRIAL,
        ControlExecutionPolicy.INSERTION_FOLLOWUP_TRIAL,
    )


def _strict_integer(payload: Mapping[str, Any], field: str) -> int:
    value = payload[field]
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"insertion rollout {field} must be an integer")
    return value


@dataclass(frozen=True)
class InsertionRolloutPosition:
    """One current step and the immutable hard cap for its rollout."""

    step_index: int
    maximum_steps: int

    def __post_init__(self) -> None:
        if (
            isinstance(self.step_index, bool)
            or not isinstance(self.step_index, int)
            or isinstance(self.maximum_steps, bool)
            or not isinstance(self.maximum_steps, int)
            or self.step_index < 1
            or self.maximum_steps
            not in tuple(
                profile.maximum_steps for profile in INSERTION_ROLLOUT_PROFILES
            )
            or self.step_index > self.maximum_steps
        ):
            raise ValueError("insertion rollout position is invalid")

    @property
    def can_followup(self) -> bool:
        return self.step_index < self.maximum_steps

    @classmethod
    def initial(cls, maximum_steps: int) -> InsertionRolloutPosition:
        return cls(1, maximum_steps)

    def followup(
        self,
        next_maximum_steps: int | None = None,
    ) -> InsertionRolloutPosition:
        if self.can_followup:
            if next_maximum_steps not in (None, self.maximum_steps):
                raise ValueError("insertion rollout cap cannot change mid-rollout")
            return InsertionRolloutPosition(
                self.step_index + 1,
                self.maximum_steps,
            )
        for previous, following in INSERTION_ROLLOUT_EXTENSIONS:
            if (
                self.step_index == previous.maximum_steps
                and self.maximum_steps == previous.maximum_steps
                and next_maximum_steps == following.maximum_steps
            ):
                return InsertionRolloutPosition(
                    self.step_index + 1,
                    following.maximum_steps,
                )
        raise ValueError("insertion rollout reached its maximum step")

    def to_dict(self) -> dict[str, int]:
        return {
            "step_index": self.step_index,
            "maximum_steps": self.maximum_steps,
        }

    @classmethod
    def from_dict(cls, payload: Any) -> InsertionRolloutPosition:
        if not isinstance(payload, Mapping):
            raise ValueError("insertion rollout position must be an object")
        try:
            return cls(
                _strict_integer(payload, "step_index"),
                _strict_integer(payload, "maximum_steps"),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("insertion rollout position is incomplete") from error


@dataclass(frozen=True)
class InsertionRolloutRoster:
    """The exact ordered session identities required by one bounded rollout."""

    session_ids: tuple[str, ...]
    maximum_steps: int

    def __post_init__(self) -> None:
        InsertionRolloutPosition.initial(self.maximum_steps)
        if (
            len(self.session_ids) != self.maximum_steps
            or len(set(self.session_ids)) != len(self.session_ids)
        ):
            raise ValueError("insertion rollout session roster is invalid")
        for session_id in self.session_ids:
            validate_safe_identifier(session_id)

    @property
    def positions(self) -> tuple[InsertionRolloutPosition, ...]:
        return tuple(
            InsertionRolloutPosition(index, self.maximum_steps)
            for index in range(1, self.maximum_steps + 1)
        )

    @classmethod
    def from_csv(
        cls,
        session_roster: str,
        maximum_steps: int,
    ) -> InsertionRolloutRoster:
        return cls(tuple(session_roster.split(",")), maximum_steps)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Print the canonical bounded insertion rollout profile."
    )
    parser.add_argument(
        "profile",
        choices=tuple(profile.name for profile in INSERTION_ROLLOUT_PROFILES),
    )
    parser.add_argument("field", choices=("maximum-steps",))
    arguments = parser.parse_args()
    print(insertion_rollout_profile(arguments.profile).maximum_steps)


if __name__ == "__main__":
    main()
