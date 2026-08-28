import json
from pathlib import Path
from types import SimpleNamespace
from tempfile import TemporaryDirectory
from typing import Optional
import unittest
from unittest.mock import patch

from jepa_wm.action import DroidAction, DroidPose
from jepa_wm.control_safety import ControlGateReason
from jepa_wm.insertion_transition import (
    INSERTION_TRANSITION_FINETUNE_SCHEMA,
    INSERTION_TRANSITION_OUTPUT_CONSTRAINT,
    InsertionProposalHandoff,
    InsertionTransitionDescendantHandoff,
    InsertionTransitionExample,
    InsertionTransitionHardExampleEvaluation,
    InsertionTransitionCandidateRank,
    InsertionTransitionSupervisionPolicy,
    InsertionTransitionTrainingSelection,
    resolve_insertion_followup_proposal,
    transition_training_examples,
    transition_hard_evaluations_fingerprint,
    transition_training_selection_fingerprint,
    validate_insertion_transition_proposal,
)
from jepa_wm.insertion_trial import InsertionTrialRollbackFailureReason
from jepa_wm.training_artifact import (
    ArtifactIdentity,
    artifact_fingerprint,
    training_report_path,
)
from jepa_wm.target_progress import (
    RealizedTargetProgressDecision,
    RealizedTargetProgressReason,
)
from sim.control_session import ControlResultStatus


class InsertionTransitionTest(unittest.TestCase):
    def test_contact_interlock_failure_becomes_pre_action_supervision(self) -> None:
        from jepa_wm.insertion_transition_evidence import (
            transition_example_from_session,
        )

        start = DroidPose((0.35, -0.27, 0.46, 0.0, 0.0, 0.0, 0.5))
        target = DroidPose((0.351, -0.27, 0.46, 0.0, 0.0, 0.0, 0.5))
        proposal = ArtifactIdentity(Path("/tmp/phase9.pth"), "a" * 64)
        summary = SimpleNamespace(
            observation=SimpleNamespace(
                observation_id=19,
                context_frame=Path("control_sessions/contact-negative/context.png"),
                target_frame=Path("recordings/held/wrist/frame_000063.png"),
                target_pose=target,
                previous_action=DroidAction((0.0,) * 7),
                warmup_frames=62,
            ),
            response=SimpleNamespace(
                proposal=proposal.path,
                proposal_fingerprint=proposal.fingerprint,
            ),
            state=SimpleNamespace(
                reference_recording="held",
                seed=42601,
                plug_attached=True,
            ),
            result=SimpleNamespace(
                status=ControlResultStatus.ROLLBACK_FAILED,
                insertion_trial_refresh=SimpleNamespace(live_pose=start),
                insertion_trial_rollback=SimpleNamespace(
                    reason=InsertionTrialRollbackFailureReason.SAFETY_INTERLOCK,
                    plug_attached=True,
                    drive_command_accepted=True,
                    interlock=SimpleNamespace(
                        maximum_contact_force_newtons=3.125,
                        collision_detected=False,
                    ),
                ),
                execution_interlock=SimpleNamespace(
                    maximum_contact_force_newtons=3.125,
                    collision_detected=False,
                ),
                post_action=None,
            ),
        )
        session = SimpleNamespace(
            session_id="contact-negative",
            direct_safety_path=Path("/missing/direct_safety.json"),
        )
        with patch(
            "sim.control_session.ControlSession.at", return_value=session
        ), patch(
            "jepa_wm.control_rollout.ControlStepSummary.from_session",
            return_value=summary,
        ):
            example = transition_example_from_session(
                Path("/tmp/data"),
                "contact-negative",
            )

        self.assertEqual(example.context_pose, start)
        self.assertEqual(example.target_pose, target)
        self.assertEqual(example.source_proposal, proposal)
        self.assertEqual(
            example.actions,
            InsertionTransitionSupervisionPolicy().actions(start, target),
        )

    def test_safe_progress_rollback_becomes_exact_live_start_supervision(self) -> None:
        from jepa_wm.insertion_transition_evidence import (
            transition_example_from_session,
        )

        start = DroidPose((0.25, -0.27, 0.46, 0.0, 0.0, 0.0, 0.5))
        target = DroidPose((0.29, -0.28, 0.45, 0.0, 0.0, 0.0, 0.5))
        proposal = ArtifactIdentity(Path("/tmp/phase3.pth"), "a" * 64)
        observation = SimpleNamespace(
            observation_id=17,
            context_frame=Path("control_sessions/progress-negative/context.png"),
            target_frame=Path("recordings/held/wrist/frame_000037.png"),
            target_pose=target,
            previous_action=DroidAction((0.0,) * 7),
            warmup_frames=36,
        )
        response = SimpleNamespace(
            proposal=proposal.path,
            proposal_fingerprint=proposal.fingerprint,
        )
        progress = RealizedTargetProgressDecision(
            0.04,
            0.0304,
            0.24,
            0.01,
            0.01,
            False,
            (RealizedTargetProgressReason.TRANSLATION_PROGRESS,),
        )
        summary = SimpleNamespace(
            observation=observation,
            response=response,
            state=SimpleNamespace(
                reference_recording="held",
                seed=42601,
                plug_attached=True,
            ),
            result=SimpleNamespace(
                status=ControlResultStatus.ROLLED_BACK_PROGRESS,
                insertion_trial_refresh=SimpleNamespace(live_pose=start),
                insertion_trial_rollback=object(),
                execution_interlock=SimpleNamespace(
                    maximum_contact_force_newtons=0.0,
                    collision_detected=False,
                ),
                post_action=SimpleNamespace(
                    plug_attached=True,
                    contact_force_newtons=0.0,
                    collision_detected=False,
                    tracking=SimpleNamespace(passed=True),
                    insertion_trial=SimpleNamespace(
                        realized_target_progress=progress,
                    ),
                ),
            ),
        )
        session = SimpleNamespace(
            session_id="progress-negative",
            direct_safety_path=Path("/missing/direct_safety.json"),
        )
        with patch(
            "sim.control_session.ControlSession.at", return_value=session
        ), patch(
            "jepa_wm.control_rollout.ControlStepSummary.from_session",
            return_value=summary,
        ):
            example = transition_example_from_session(
                Path("/tmp/data"),
                "progress-negative",
            )

        self.assertEqual(example.context_pose, start)
        self.assertEqual(example.target_pose, target)
        self.assertEqual(example.source_proposal, proposal)
        self.assertEqual(
            example.actions,
            InsertionTransitionSupervisionPolicy().actions(start, target),
        )

    def test_descendant_may_be_trained_from_the_live_predecessor_proposal(self) -> None:
        parent = ArtifactIdentity(Path("/tmp/parent.pth"), "a" * 64)
        descendant = ArtifactIdentity(Path("/tmp/child.pth"), "b" * 64)

        handoff = InsertionTransitionDescendantHandoff(
            parent,
            parent,
            descendant,
            "training-source",
        )

        self.assertEqual(handoff.requested, descendant)

    def test_transition_head_holds_rotation_and_gripper_for_every_input(self) -> None:
        try:
            import torch
        except ModuleNotFoundError:
            self.skipTest("PyTorch is unavailable in the local test runtime")
        from jepa_wm.insertion_transition_finetune import (
            constrain_insertion_transition_output,
        )
        from jepa_wm.proposal import ActionProposalNetwork, ProposalFeatureMode

        proposal = ActionProposalNetwork(
            feature_dimension=4,
            horizon=3,
            hidden_dimension=8,
            action_mean=torch.tensor([[0.1, -0.1, 0.2, 0.03, -0.04, 0.05, 0.2]] * 3),
            action_standard_deviation=torch.ones((3, 7)),
            feature_mode=ProposalFeatureMode.GLOBAL,
        )
        constrain_insertion_transition_output(proposal)

        actions = proposal(torch.randn(2, 5, 4), torch.randn(2, 5, 4))

        self.assertTrue(torch.equal(actions[:, :, 3:], torch.zeros(2, 3, 4)))

    def test_hard_objective_combines_mse_and_goal_direction_on_cpu(self) -> None:
        try:
            import torch
        except ModuleNotFoundError:
            self.skipTest("PyTorch is unavailable in the local test runtime")
        from jepa_wm.insertion_transition_finetune import (
            InsertionTransitionHardObjective,
        )

        predicted = torch.zeros((1, 3, 7))
        predicted[0, 0, 0] = 1.0
        objective = InsertionTransitionHardObjective.from_predictions(
            predicted,
            torch.zeros_like(predicted),
            torch.zeros((3, 7)),
            torch.ones((3, 7)),
            torch.tensor([[-1.0, 0.0, 0.0]]),
            1.0,
        )

        self.assertAlmostEqual(float(objective.hard_loss), 1.0 / 21.0, delta=1e-6)
        self.assertAlmostEqual(float(objective.direction_loss), 2.0, delta=1e-6)
        self.assertAlmostEqual(
            float(objective.total),
            2.0 + 1.0 / 21.0,
            delta=1e-6,
        )

    def example(self) -> InsertionTransitionExample:
        start = DroidPose((0.31, -0.29, 0.45, 0.0, 0.0, 0.0, 0.5))
        target = DroidPose((0.30, -0.295, 0.47, 0.0, 0.0, 0.0, 0.5))
        policy = InsertionTransitionSupervisionPolicy()
        return InsertionTransitionExample(
            source_session_id="transition-source",
            reference_recording="contact-insertion-held-00",
            seed=42600,
            observation_id=7,
            context_frame=Path("control_sessions/transition-source/context.png"),
            target_frame=Path(
                "recordings/contact-insertion-held-00/wrist/frame_000024.png"
            ),
            context_pose=start,
            target_pose=target,
            previous_action=DroidAction((0.0,) * 7),
            task_context_index=21,
            source_proposal=ArtifactIdentity(Path("/tmp/proposal.pth"), "a" * 64),
            actions=policy.actions(start, target),
            supervision=policy,
        )

    def hard_evaluation(
        self,
        example: InsertionTransitionExample,
        *,
        reason: Optional[ControlGateReason] = None,
    ) -> InsertionTransitionHardExampleEvaluation:
        return InsertionTransitionHardExampleEvaluation(
            source_session_id=example.source_session_id,
            first_action_goal_cosine=1.0,
            predicted_actions=example.actions,
            failure_reason=reason,
        )

    def inadmissible_example(self) -> InsertionTransitionExample:
        example = self.example()
        target = DroidPose(
            (
                example.context_pose.values[0] + 0.051,
                *example.context_pose.values[1:],
            )
        )
        return InsertionTransitionExample.from_dict(
            {
                **example.to_dict(),
                "source_session_id": "inadmissible-transition-source",
                "reference_recording": "contact-insertion-held-01",
                "target_pose": list(target.values),
                "actions": [
                    list(action.values)
                    for action in example.supervision.actions(
                        example.context_pose,
                        target,
                    )
                ],
            }
        )

    def test_training_selection_retains_but_does_not_rehearse_inadmissible_targets(
        self,
    ) -> None:
        current = self.example()
        parent = current.source_proposal
        admissible = InsertionTransitionExample.from_dict(
            {
                **current.to_dict(),
                "source_session_id": "admissible-ancestor",
            }
        )
        inadmissible = self.inadmissible_example()
        self.assertTrue(admissible.target_progress_admissible)
        self.assertFalse(inadmissible.target_progress_admissible)
        selection = InsertionTransitionTrainingSelection(
            parent=parent,
            transition_example=current,
            evaluation_exclusions=(
                admissible.reference_recording,
                inadmissible.reference_recording,
            ),
            rehearsal_recordings=("contact-insertion-train-00",),
            rehearsal_context_indices=(3, 11),
            rehearsal_transition_examples=(admissible, inadmissible),
        )

        selection.validate_rehearsal((admissible, inadmissible))
        self.assertEqual(
            selection.actionable_rehearsal_transition_examples,
            (admissible,),
        )
        self.assertEqual(
            InsertionTransitionTrainingSelection.from_dict(selection.to_dict()),
            selection,
        )
        with self.assertRaisesRegex(ValueError, "rehearsal provenance"):
            selection.validate_rehearsal((inadmissible,))

    def test_hard_evaluation_round_trip_rejects_coerced_or_contradictory_claims(
        self,
    ) -> None:
        evaluation = self.hard_evaluation(self.example())
        self.assertEqual(
            InsertionTransitionHardExampleEvaluation.from_dict(evaluation.to_dict()),
            evaluation,
        )

        payload = evaluation.to_dict()
        payload["passed"] = False
        with self.assertRaisesRegex(ValueError, "hard evaluation"):
            InsertionTransitionHardExampleEvaluation.from_dict(payload)

        payload = evaluation.to_dict()
        payload["first_action_goal_cosine"] = True
        with self.assertRaisesRegex(ValueError, "hard evaluation"):
            InsertionTransitionHardExampleEvaluation.from_dict(payload)

        payload = evaluation.to_dict()
        payload["source_session_id"] = 17
        with self.assertRaisesRegex(ValueError, "hard evaluation"):
            InsertionTransitionHardExampleEvaluation.from_dict(payload)

        substituted_cosine = InsertionTransitionHardExampleEvaluation(
            source_session_id=evaluation.source_session_id,
            first_action_goal_cosine=0.5,
            predicted_actions=evaluation.predicted_actions,
            failure_reason=evaluation.failure_reason,
        )
        with self.assertRaisesRegex(ValueError, "not bound"):
            substituted_cosine.validate_example(self.example())

        reconstructed = InsertionTransitionHardExampleEvaluation.from_prediction(
            self.example(),
            evaluation.predicted_actions,
        )
        self.assertEqual(reconstructed.first_action_goal_cosine, 1.0)
        self.assertTrue(reconstructed.passed)

        oversized = InsertionTransitionHardExampleEvaluation.from_prediction(
            self.example(),
            (
                DroidAction(
                    (
                        *(
                            value * 3.0
                            for value in evaluation.predicted_actions[0].values[:3]
                        ),
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    )
                ),
                *evaluation.predicted_actions[1:],
            ),
        )
        self.assertFalse(oversized.passed)
        self.assertIs(
            oversized.failure_reason,
            ControlGateReason.ACTION_OUT_OF_BOUNDS,
        )
        rotated_action = DroidAction(
            (
                *evaluation.predicted_actions[0].values[:3],
                1e-3,
                0.0,
                0.0,
                0.0,
            )
        )
        rotated = InsertionTransitionHardExampleEvaluation.from_prediction(
            self.example(),
            (rotated_action, *evaluation.predicted_actions[1:]),
        )
        self.assertIs(rotated.failure_reason, ControlGateReason.ACTION_OUT_OF_BOUNDS)

    def test_candidate_rank_prioritizes_exact_gate_acceptance_before_loss(self) -> None:
        passing = InsertionTransitionCandidateRank.from_evaluations(
            (self.hard_evaluation(self.example()),),
            hard_objective=2.0,
        )
        failing = InsertionTransitionCandidateRank.from_evaluations(
            (
                self.hard_evaluation(
                    self.example(),
                    reason=ControlGateReason.TARGET_PROGRESS_INSUFFICIENT,
                ),
            ),
            hard_objective=0.01,
        )
        lower_loss_passing = InsertionTransitionCandidateRank.from_evaluations(
            (self.hard_evaluation(self.example()),),
            hard_objective=1.0,
        )

        self.assertLess(passing, failing)
        self.assertLess(lower_loss_passing, passing)

    def test_supervision_splits_translation_and_holds_other_axes(self) -> None:
        example = self.example()
        self.assertEqual(len(example.actions), 3)
        self.assertAlmostEqual(example.actions[0].values[0], -0.01 / 3.0)
        self.assertAlmostEqual(example.actions[0].values[1], -0.005 / 3.0)
        self.assertAlmostEqual(example.actions[0].values[2], 0.02 / 3.0)
        self.assertEqual(example.actions[0].values[3:], (0.0, 0.0, 0.0, 0.0))
        self.assertGreater(
            sum(
                action * goal
                for action, goal in zip(
                    example.actions[0].values[:3],
                    (
                        example.target_pose.values[index]
                        - example.context_pose.values[index]
                        for index in range(3)
                    ),
                )
            ),
            0.0,
        )

    def test_current_supervision_stays_inside_the_full_scale_safety_margin(
        self,
    ) -> None:
        start = DroidPose((0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5))
        target = DroidPose((0.06, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5))
        policy = InsertionTransitionSupervisionPolicy()

        actions = policy.actions(start, target)

        self.assertAlmostEqual(
            sum(value * value for value in actions[0].values[:3]) ** 0.5,
            0.012,
        )
        self.assertEqual(
            InsertionTransitionSupervisionPolicy.from_dict(policy.to_dict()),
            policy,
        )
        legacy = InsertionTransitionSupervisionPolicy.from_dict({"action_horizon": 3})
        self.assertAlmostEqual(
            sum(value * value for value in legacy.actions(start, target)[0].values[:3])
            ** 0.5,
            0.02,
        )

    def test_round_trip_and_rollout_bind_exact_frames(self) -> None:
        example = self.example()
        self.assertEqual(
            InsertionTransitionExample.from_dict(example.to_dict()), example
        )
        with TemporaryDirectory() as directory:
            root = Path(directory)
            for relative in (example.context_frame, example.target_frame):
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"image")
            rollout = example.rollout(root)
        self.assertEqual(rollout.context_pose, example.context_pose)
        self.assertEqual(rollout.target_pose, example.target_pose)
        self.assertEqual(rollout.actions, example.actions)
        self.assertEqual(rollout.target.index, 24)

    def test_raw_policy_and_tampered_actions_fail_closed(self) -> None:
        example = self.example()
        payload = example.to_dict()
        payload["supervision"]["action_horizon"] = True
        with self.assertRaisesRegex(ValueError, "incomplete"):
            InsertionTransitionExample.from_dict(payload)

        payload = example.to_dict()
        payload["actions"][0][0] *= -1.0
        with self.assertRaisesRegex(ValueError, "incomplete"):
            InsertionTransitionExample.from_dict(payload)

    @patch("jepa_wm.insertion_proposal_readiness.validate_insertion_proposal")
    @patch("jepa_wm.insertion_transition._checkpoint_training_fingerprints")
    def test_only_exact_constrained_bridge_parent_can_continue_rollout(
        self,
        checkpoint_fingerprints,
        validate_base_proposal,
    ) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            bridge = root / "bridge.pth"
            parent = root / "parent.pth"
            other = root / "other.pth"
            bridge.write_bytes(b"bridge")
            parent.write_bytes(b"parent")
            other.write_bytes(b"other")
            parent_identity = ArtifactIdentity.from_artifact(parent)
            validate_base_proposal.return_value = SimpleNamespace(
                identity=parent_identity
            )
            example = self.example()
            transition_example = InsertionTransitionExample.from_dict(
                {
                    **example.to_dict(),
                    "source_proposal": parent_identity.to_dict(),
                }
            )
            selection = InsertionTransitionTrainingSelection(
                parent=parent_identity,
                transition_example=transition_example,
                evaluation_exclusions=(example.reference_recording,),
                rehearsal_recordings=("contact-insertion-train-00",),
                rehearsal_context_indices=(3, 11),
            ).to_dict()
            training_report_path(bridge).write_text(
                json.dumps(
                    {
                        "schema": INSERTION_TRANSITION_FINETUNE_SCHEMA,
                        "status": "trained",
                        "proposal": str(bridge.resolve()),
                        "proposal_fingerprint": artifact_fingerprint(bridge),
                        "output_constraint": INSERTION_TRANSITION_OUTPUT_CONSTRAINT,
                        "training_selection": selection,
                        "training_selection_fingerprint": (
                            transition_training_selection_fingerprint(selection)
                        ),
                    }
                )
            )
            checkpoint_fingerprints.return_value = SimpleNamespace(
                selection=transition_training_selection_fingerprint(selection),
                evaluation=None,
            )

            self.assertEqual(
                resolve_insertion_followup_proposal(bridge, parent),
                parent.resolve(),
            )
            handoff = InsertionProposalHandoff.from_bridge(bridge, parent)
            self.assertEqual(
                validate_insertion_transition_proposal(bridge),
                ArtifactIdentity.from_artifact(bridge),
            )
            self.assertEqual(
                InsertionProposalHandoff.from_dict(handoff.to_dict()),
                handoff,
            )
            self.assertEqual(handoff.previous, handoff.bridge)
            self.assertEqual(
                handoff.resolve(bridge, artifact_fingerprint(bridge), parent),
                parent.resolve(),
            )
            with self.assertRaisesRegex(ValueError, "bridge response"):
                handoff.resolve(bridge, "f" * 64, parent)
            descendant = InsertionTransitionDescendantHandoff(
                ArtifactIdentity.from_artifact(bridge),
                parent_identity,
                ArtifactIdentity.from_artifact(other),
                "phase2-safety-source",
            )
            self.assertEqual(
                InsertionTransitionDescendantHandoff.from_dict(descendant.to_dict()),
                descendant,
            )
            self.assertEqual(
                descendant.previous,
                ArtifactIdentity.from_artifact(bridge),
            )
            self.assertEqual(
                descendant.resolve(
                    bridge,
                    artifact_fingerprint(bridge),
                    other,
                ),
                other.resolve(),
            )
            self.assertEqual(
                resolve_insertion_followup_proposal(parent, parent),
                parent.resolve(),
            )
            with self.assertRaisesRegex(ValueError, "exact frozen parent"):
                resolve_insertion_followup_proposal(bridge, other)

            payload = json.loads(training_report_path(bridge).read_text())
            payload["training_selection"]["parent"] = ArtifactIdentity.from_artifact(
                other
            ).to_dict()
            training_report_path(bridge).write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError, "provenance|fingerprint"):
                resolve_insertion_followup_proposal(bridge, parent)

    @patch("jepa_wm.insertion_proposal_readiness.validate_insertion_proposal")
    @patch("jepa_wm.insertion_transition._checkpoint_training_fingerprints")
    def test_descendant_rehearses_every_authenticated_transition_example(
        self,
        checkpoint_fingerprints,
        validate_base_proposal,
    ) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            base = root / "base.pth"
            phase1 = root / "phase1.pth"
            phase2 = root / "phase2.pth"
            for path in (base, phase1, phase2):
                path.write_bytes(path.stem.encode())

            base_identity = ArtifactIdentity.from_artifact(base)
            phase1_identity = ArtifactIdentity.from_artifact(phase1)
            validate_base_proposal.return_value = SimpleNamespace(
                identity=base_identity
            )
            first = InsertionTransitionExample.from_dict(
                {
                    **self.example().to_dict(),
                    "source_session_id": "phase1-source",
                    "source_proposal": base_identity.to_dict(),
                }
            )
            first_selection = InsertionTransitionTrainingSelection(
                parent=base_identity,
                transition_example=first,
                evaluation_exclusions=(first.reference_recording,),
                rehearsal_recordings=("contact-insertion-train-00",),
                rehearsal_context_indices=(3, 11),
            ).to_dict()
            training_report_path(phase1).write_text(
                json.dumps(
                    {
                        "schema": INSERTION_TRANSITION_FINETUNE_SCHEMA,
                        "status": "trained",
                        "proposal": str(phase1.resolve()),
                        "proposal_fingerprint": artifact_fingerprint(phase1),
                        "output_constraint": INSERTION_TRANSITION_OUTPUT_CONSTRAINT,
                        "training_selection": first_selection,
                        "training_selection_fingerprint": (
                            transition_training_selection_fingerprint(first_selection)
                        ),
                    }
                )
            )

            second = InsertionTransitionExample.from_dict(
                {
                    **self.example().to_dict(),
                    "source_session_id": "phase2-source",
                    "observation_id": 8,
                    "source_proposal": phase1_identity.to_dict(),
                }
            )
            second_selection = InsertionTransitionTrainingSelection(
                parent=phase1_identity,
                transition_example=second,
                evaluation_exclusions=(first.reference_recording,),
                rehearsal_recordings=("contact-insertion-train-00",),
                rehearsal_context_indices=(3, 11),
                rehearsal_transition_examples=(first,),
            ).to_dict()
            second_evaluations = (
                self.hard_evaluation(second),
                self.hard_evaluation(first),
            )
            second_evaluation_fingerprint = transition_hard_evaluations_fingerprint(
                second_evaluations
            )
            training_report_path(phase2).write_text(
                json.dumps(
                    {
                        "schema": INSERTION_TRANSITION_FINETUNE_SCHEMA,
                        "status": "trained",
                        "proposal": str(phase2.resolve()),
                        "proposal_fingerprint": artifact_fingerprint(phase2),
                        "output_constraint": INSERTION_TRANSITION_OUTPUT_CONSTRAINT,
                        "training_selection": second_selection,
                        "training_selection_fingerprint": (
                            transition_training_selection_fingerprint(second_selection)
                        ),
                        "hard_example_evaluations": [
                            evaluation.to_dict() for evaluation in second_evaluations
                        ],
                        "hard_example_evaluations_fingerprint": (
                            second_evaluation_fingerprint
                        ),
                    }
                )
            )
            first_selection_fingerprint = transition_training_selection_fingerprint(
                first_selection
            )
            second_selection_fingerprint = transition_training_selection_fingerprint(
                second_selection
            )
            checkpoint_fingerprints.side_effect = lambda path: SimpleNamespace(
                selection=(
                    first_selection_fingerprint
                    if path.resolve() == phase1.resolve()
                    else second_selection_fingerprint
                ),
                evaluation=(
                    second_evaluation_fingerprint
                    if path.resolve() == phase2.resolve()
                    else None
                ),
            )

            self.assertEqual(
                transition_training_examples(phase2),
                (first, second),
            )

            substituted = InsertionTransitionHardExampleEvaluation(
                source_session_id=second.source_session_id,
                first_action_goal_cosine=0.0,
                predicted_actions=(DroidAction((0.0,) * 7), *second.actions[1:]),
                failure_reason=(ControlGateReason.TARGET_PROGRESS_INSUFFICIENT),
            )
            substituted_evaluations = (substituted, second_evaluations[1])
            substituted_payload = json.loads(training_report_path(phase2).read_text())
            substituted_payload["hard_example_evaluations"] = [
                evaluation.to_dict() for evaluation in substituted_evaluations
            ]
            substituted_payload["hard_example_evaluations_fingerprint"] = (
                transition_hard_evaluations_fingerprint(substituted_evaluations)
            )
            training_report_path(phase2).write_text(json.dumps(substituted_payload))
            with self.assertRaisesRegex(ValueError, "evaluation fingerprint"):
                transition_training_examples(phase2)

            training_report_path(phase2).write_text(
                json.dumps(
                    {
                        **substituted_payload,
                        "hard_example_evaluations": [
                            evaluation.to_dict() for evaluation in second_evaluations
                        ],
                        "hard_example_evaluations_fingerprint": (
                            second_evaluation_fingerprint
                        ),
                    }
                )
            )
            second_selection["rehearsal_transition_examples"] = []
            training_report_path(phase2).write_text(
                json.dumps(
                    {
                        "schema": INSERTION_TRANSITION_FINETUNE_SCHEMA,
                        "status": "trained",
                        "proposal": str(phase2.resolve()),
                        "proposal_fingerprint": artifact_fingerprint(phase2),
                        "output_constraint": INSERTION_TRANSITION_OUTPUT_CONSTRAINT,
                        "training_selection": second_selection,
                        "training_selection_fingerprint": (
                            transition_training_selection_fingerprint(second_selection)
                        ),
                    }
                )
            )
            with self.assertRaisesRegex(ValueError, "fingerprint"):
                transition_training_examples(phase2)

            training_report_path(base).write_text(
                json.dumps({"schema": "mutated-non-transition-sidecar"})
            )
            validate_base_proposal.side_effect = ValueError(
                "base proposal authentication failed"
            )
            with self.assertRaisesRegex(ValueError, "base proposal"):
                transition_training_examples(phase1)


if __name__ == "__main__":
    unittest.main()
