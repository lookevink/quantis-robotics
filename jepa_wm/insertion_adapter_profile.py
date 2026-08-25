"""Canonical insertion adapter profiles shared by training and planning."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Sequence

from jepa_wm.action_activity import DroidActionActivityThresholds
from jepa_wm.candidate_policy import CandidateNoisePolicy

if TYPE_CHECKING:
    from jepa_wm.candidate_negatives import CandidateMiningConfig


@dataclass(frozen=True)
class InsertionAdapterProfileDescriptor:
    artifact_stem: str
    minimum_goal_cosine: float | None
    first_action_activity: DroidActionActivityThresholds = (
        DroidActionActivityThresholds()
    )
    noise_policy: CandidateNoisePolicy = CandidateNoisePolicy()
    learning_rate: float = 1e-3
    initial_profile: InsertionAdapterProfile | None = None
    required_training_steps: int | None = None

    def candidate_mining_config(self) -> CandidateMiningConfig:
        from jepa_wm.candidate_negatives import CandidateMiningConfig

        return CandidateMiningConfig(
            minimum_goal_cosine=self.minimum_goal_cosine,
            first_action_activity=self.first_action_activity,
            noise_policy=self.noise_policy,
        )

    def initial_adapter_path(self, output: Path, steps: int) -> Path:
        if self.initial_profile is None:
            raise ValueError("adapter profile does not use an initial adapter")
        if self.required_training_steps is not None and (
            steps != self.required_training_steps
        ):
            raise ValueError("adapter profile requires one exact training epoch")
        suffix = f"{self.artifact_stem}{steps}.pth"
        if not output.name.endswith(suffix):
            raise ValueError("adapter output does not match its profile and steps")
        prefix = output.name[: -len(suffix)]
        initial_stem = self.initial_profile.descriptor.artifact_stem
        return output.with_name(f"{prefix}{initial_stem}{steps}.pth").resolve()


class InsertionAdapterProfile(str, Enum):
    GENERIC = "generic"
    GOAL_ALIGNED = "goal_aligned"
    GOAL_ALIGNED_RELATIVE = "goal_aligned_relative"
    GOAL_ALIGNED_RELATIVE_FINETUNE = "goal_aligned_relative_finetune"

    @property
    def descriptor(self) -> InsertionAdapterProfileDescriptor:
        return INSERTION_ADAPTER_PROFILES[self]


GOAL_ALIGNED_MINIMUM_COSINE = 0.95
GOAL_ALIGNED_ACTIVITY = DroidActionActivityThresholds(
    translation_norm=1e-5,
    rotation_norm=1e-5,
    gripper_delta=0.005,
)
GOAL_ALIGNED_RELATIVE_NOISE = CandidateNoisePolicy.recorded_action(
    translation_floor=1e-5,
    rotation_floor=1e-5,
    gripper_floor=0.005,
)


INSERTION_ADAPTER_PROFILES = {
    InsertionAdapterProfile.GENERIC: InsertionAdapterProfileDescriptor(
        artifact_stem="insertion_adapter_s",
        minimum_goal_cosine=None,
    ),
    InsertionAdapterProfile.GOAL_ALIGNED: InsertionAdapterProfileDescriptor(
        artifact_stem="insertion_adapter_goal_aligned_s",
        minimum_goal_cosine=GOAL_ALIGNED_MINIMUM_COSINE,
        first_action_activity=GOAL_ALIGNED_ACTIVITY,
    ),
    InsertionAdapterProfile.GOAL_ALIGNED_RELATIVE: InsertionAdapterProfileDescriptor(
        artifact_stem="insertion_adapter_goal_aligned_relative_s",
        minimum_goal_cosine=GOAL_ALIGNED_MINIMUM_COSINE,
        first_action_activity=GOAL_ALIGNED_ACTIVITY,
        noise_policy=GOAL_ALIGNED_RELATIVE_NOISE,
    ),
    InsertionAdapterProfile.GOAL_ALIGNED_RELATIVE_FINETUNE: (
        InsertionAdapterProfileDescriptor(
            artifact_stem="insertion_adapter_goal_aligned_relative_finetune_s",
            minimum_goal_cosine=GOAL_ALIGNED_MINIMUM_COSINE,
            first_action_activity=GOAL_ALIGNED_ACTIVITY,
            noise_policy=GOAL_ALIGNED_RELATIVE_NOISE,
            learning_rate=1e-4,
            initial_profile=InsertionAdapterProfile.GENERIC,
            required_training_steps=1056,
        )
    ),
}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "profile",
        choices=tuple(profile.value for profile in InsertionAdapterProfile),
    )
    parser.add_argument("field", choices=("artifact-stem", "minimum-goal-cosine"))
    args = parser.parse_args(argv)
    descriptor = InsertionAdapterProfile(args.profile).descriptor
    value = (
        descriptor.artifact_stem
        if args.field == "artifact-stem"
        else descriptor.minimum_goal_cosine
    )
    print("" if value is None else value)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
