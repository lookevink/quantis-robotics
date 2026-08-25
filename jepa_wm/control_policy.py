"""Typed execution-policy vocabulary shared by control producers and Isaac."""

from enum import Enum


class ControlExecutionPolicy(str, Enum):
    DIRECT = "direct"
    CALIBRATION_COLLECTION = "calibration_collection"
    INSERTION_SAFETY_EVALUATION = "insertion_safety_evaluation"
    INSERTION_RESET_TRIAL = "insertion_reset_trial"
    ZERO_BASELINE = "zero"
    SCRIPTED_BASELINE = "scripted"
    RESET_TRIAL_CANDIDATE = "reset_trial_candidate"
