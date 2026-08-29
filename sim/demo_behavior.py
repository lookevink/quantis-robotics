"""Current executable behavior frozen into one bounded demo experiment."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from jepa_wm.action import ACTION_RECORDING_CONTRACT
from jepa_wm.contact_grasp_target import CONTACT_GRASP_TARGET_POLICY
from jepa_wm.control_policy import ControlExecutionPolicy
from jepa_wm.control_safety import SimulatorSafetyLimits
from jepa_wm.control_tracking import tracking_limits_for_policy
from jepa_wm.grasp_task import GraspTaskLimits, MAXIMUM_CONTACT_GRASP_ACTIONS
from jepa_wm.grasp_to_insertion import GRASP_TO_INSERTION_SCHEMA
from jepa_wm.insertion_contract import insertion_control_target_policy
from jepa_wm.insertion_rollout import DEMO_INSERTION_ROLLOUT
from jepa_wm.insertion_task import InsertionTaskLimits
from jepa_wm.insertion_trial import InsertionTrialPolicy
from jepa_wm.replay_verification import ReplayLimits
from jepa_wm.task_windows import CONTACT_GRASP_PROPOSAL_WINDOW
from jepa_wm.training_artifact import validate_artifact_fingerprint
from sim.control_capture_schedule import (
    CONTROL_KNOWN_START_SCHEMA,
    KNOWN_START_ORIENTATION_TOLERANCE_RADIANS,
    KNOWN_START_POSITION_TOLERANCE_METERS,
    canonical_control_fingerprint,
    control_capture_schedule,
)
from sim.control_context import ControlContextPurpose
from sim.isaac_demo_camera import (
    JEPA_WM_CAMERA_SPECS,
    WRIST_CAMERA_TARGET_METERS,
    WRIST_CAMERA_TRANSLATION_METERS,
    WRIST_CAMERA_UP_AXIS,
)


@dataclass(frozen=True)
class DemoTerminalContract:
    grasp_actions: int
    insertion_actions: int
    require_seated_hold: bool

    def __post_init__(self) -> None:
        if (
            isinstance(self.grasp_actions, bool)
            or not isinstance(self.grasp_actions, int)
            or self.grasp_actions <= 0
            or isinstance(self.insertion_actions, bool)
            or not isinstance(self.insertion_actions, int)
            or self.insertion_actions <= 0
            or not isinstance(self.require_seated_hold, bool)
        ):
            raise ValueError("demo terminal contract is invalid")

    @property
    def action_cap(self) -> int:
        return self.grasp_actions + self.insertion_actions

    @classmethod
    def from_dict(cls, payload: Any) -> DemoTerminalContract:
        if not isinstance(payload, dict) or set(payload) != {
            "grasp_actions",
            "insertion_actions",
            "require_seated_hold",
        }:
            raise ValueError("demo terminal contract payload is invalid")
        return cls(
            payload["grasp_actions"],
            payload["insertion_actions"],
            payload["require_seated_hold"],
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "grasp_actions": self.grasp_actions,
            "insertion_actions": self.insertion_actions,
            "require_seated_hold": self.require_seated_hold,
        }


@dataclass(frozen=True)
class DemoBehavioralContract:
    """Fingerprinted projection of every current live-behavior authority."""

    execution_policy: ControlExecutionPolicy
    control_policy_fingerprint: str
    safety_limits_fingerprint: str
    reset_contract_fingerprint: str
    schedule_fingerprint: str
    evidence_schema: str
    camera_configuration_fingerprint: str
    terminal: DemoTerminalContract

    def __post_init__(self) -> None:
        try:
            if not isinstance(self.execution_policy, ControlExecutionPolicy):
                raise ValueError
            for fingerprint in (
                self.control_policy_fingerprint,
                self.safety_limits_fingerprint,
                self.reset_contract_fingerprint,
                self.schedule_fingerprint,
                self.camera_configuration_fingerprint,
            ):
                validate_artifact_fingerprint(fingerprint)
        except ValueError as error:
            raise ValueError("demo behavioral contract is invalid") from error
        if not self.evidence_schema.startswith("quantis."):
            raise ValueError("demo behavioral contract is invalid")

    @property
    def action_cap(self) -> int:
        return self.terminal.action_cap

    @classmethod
    def from_dict(cls, payload: Any) -> DemoBehavioralContract:
        fields = {
            "execution_policy",
            "control_policy_fingerprint",
            "safety_limits_fingerprint",
            "reset_contract_fingerprint",
            "schedule_fingerprint",
            "evidence_schema",
            "camera_configuration_fingerprint",
            "action_cap",
            "terminal",
        }
        if not isinstance(payload, dict) or set(payload) != fields:
            raise ValueError("demo behavioral contract payload is invalid")
        try:
            contract = cls(
                execution_policy=ControlExecutionPolicy(payload["execution_policy"]),
                control_policy_fingerprint=str(payload["control_policy_fingerprint"]),
                safety_limits_fingerprint=str(payload["safety_limits_fingerprint"]),
                reset_contract_fingerprint=str(payload["reset_contract_fingerprint"]),
                schedule_fingerprint=str(payload["schedule_fingerprint"]),
                evidence_schema=str(payload["evidence_schema"]),
                camera_configuration_fingerprint=str(
                    payload["camera_configuration_fingerprint"]
                ),
                terminal=DemoTerminalContract.from_dict(payload["terminal"]),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("demo behavioral contract payload is invalid") from error
        if payload["action_cap"] != contract.action_cap:
            raise ValueError("demo behavioral action cap is inconsistent")
        return contract

    def to_dict(self) -> dict[str, Any]:
        return {
            "execution_policy": self.execution_policy.value,
            "control_policy_fingerprint": self.control_policy_fingerprint,
            "safety_limits_fingerprint": self.safety_limits_fingerprint,
            "reset_contract_fingerprint": self.reset_contract_fingerprint,
            "schedule_fingerprint": self.schedule_fingerprint,
            "evidence_schema": self.evidence_schema,
            "camera_configuration_fingerprint": (self.camera_configuration_fingerprint),
            "action_cap": self.action_cap,
            "terminal": self.terminal.to_dict(),
        }


def current_demo_behavioral_contract() -> DemoBehavioralContract:
    """Derive the frozen contract from the code that a live run will execute."""

    grasp_policy = ControlExecutionPolicy.DIRECT
    insertion_policy = ControlExecutionPolicy.INSERTION_FOLLOWUP_TRIAL
    insertion_target = insertion_control_target_policy(insertion_policy)
    if insertion_target is None:
        raise RuntimeError("demo insertion target policy is missing")
    schedule = control_capture_schedule(
        grasp_policy,
        insertion_control=True,
        context_index=CONTACT_GRASP_PROPOSAL_WINDOW.start_index,
        context_purpose=ControlContextPurpose.CONTACT_GRASP,
    )
    return DemoBehavioralContract(
        execution_policy=grasp_policy,
        control_policy_fingerprint=canonical_control_fingerprint(
            {
                "grasp_execution_policy": grasp_policy.value,
                "grasp_target_policy": CONTACT_GRASP_TARGET_POLICY.to_dict(),
                "insertion_execution_policy": insertion_policy.value,
                "insertion_target_policy": insertion_target.to_dict(),
                "insertion_trial_policy": InsertionTrialPolicy().to_dict(),
                "action_recording_contract": ACTION_RECORDING_CONTRACT.to_dict(),
            }
        ),
        safety_limits_fingerprint=canonical_control_fingerprint(
            {
                "simulator": SimulatorSafetyLimits().to_dict(),
                "grasp_task": asdict(GraspTaskLimits()),
                "insertion_task": asdict(InsertionTaskLimits()),
                "grasp_tracking": asdict(tracking_limits_for_policy(grasp_policy)),
                "insertion_tracking": asdict(
                    tracking_limits_for_policy(insertion_policy)
                ),
                "replay": ReplayLimits().to_dict(),
            }
        ),
        reset_contract_fingerprint=canonical_control_fingerprint(
            {
                "schema": CONTROL_KNOWN_START_SCHEMA,
                "context_index": CONTACT_GRASP_PROPOSAL_WINDOW.start_index,
                "context_purpose": ControlContextPurpose.CONTACT_GRASP.value,
                "position_tolerance_meters": (KNOWN_START_POSITION_TOLERANCE_METERS),
                "orientation_tolerance_radians": (
                    KNOWN_START_ORIENTATION_TOLERANCE_RADIANS
                ),
                "direct_state_setting": "reset_and_initialization_only",
                "runtime_motion": "drive_only",
            }
        ),
        schedule_fingerprint=schedule.fingerprint,
        evidence_schema=GRASP_TO_INSERTION_SCHEMA,
        camera_configuration_fingerprint=canonical_control_fingerprint(
            {
                "control_cameras": [
                    {
                        "label": camera.label,
                        "path": camera.path,
                        "resolution": list(camera.resolution),
                    }
                    for camera in JEPA_WM_CAMERA_SPECS
                ],
                "wrist_mount": {
                    "translation_meters": list(WRIST_CAMERA_TRANSLATION_METERS),
                    "target_meters": list(WRIST_CAMERA_TARGET_METERS),
                    "up_axis": list(WRIST_CAMERA_UP_AXIS),
                },
                "presentation_recording": "external_not_managed",
            }
        ),
        terminal=DemoTerminalContract(
            MAXIMUM_CONTACT_GRASP_ACTIONS,
            DEMO_INSERTION_ROLLOUT.maximum_steps,
            False,
        ),
    )
