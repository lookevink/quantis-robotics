"""Send one versioned observation to the resident Unix-socket worker."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import socket
from typing import TYPE_CHECKING

from jepa_wm.control_protocol import ControlObservation, ProposedControl
from jepa_wm.persistence import write_json_atomic
from jepa_wm.objective_calibration import (
    ActionResponseCalibration,
    CalibrationIdentity,
)
from jepa_wm.worker_artifacts import ControlWorkerArtifacts
from jepa_wm.trajectory import validate_observation_target
from sim.control_session import ControlSessionState

if TYPE_CHECKING:
    from jepa_wm.shadow_planning import ShadowPlanningRequest, ShadowSearchEvidence


def _request(socket_path: Path, payload: dict) -> dict:
    request = json.dumps(payload, separators=(",", ":")) + "\n"
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
        client.connect(str(socket_path))
        client.sendall(request.encode("utf-8"))
        client.shutdown(socket.SHUT_WR)
        response = b""
        while True:
            chunk = client.recv(65536)
            if not chunk:
                break
            response += chunk
    lines = response.decode("utf-8").splitlines()
    if len(lines) != 1:
        raise ValueError("control worker returned an invalid response count")
    return json.loads(lines[0])


def request_control(socket_path: Path, observation: ControlObservation) -> ProposedControl:
    return ProposedControl.from_dict(_request(socket_path, observation.to_dict()))


def request_shadow_plan(
    socket_path: Path,
    request: ShadowPlanningRequest,
) -> ShadowSearchEvidence:
    from jepa_wm.shadow_planning import ShadowSearchEvidence

    return ShadowSearchEvidence.from_dict(_request(socket_path, request.to_dict()))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--socket", type=Path, required=True)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--direct-response", type=Path)
    parser.add_argument("--artifacts", type=Path)
    parser.add_argument("--shadow-request-output", type=Path)
    parser.add_argument("--shadow-response-output", type=Path)
    parser.add_argument("--state", type=Path)
    parser.add_argument("--recording-root", type=Path)
    args = parser.parse_args()
    observation = ControlObservation.from_dict(json.loads(args.request.read_text()))
    if args.direct_response is None:
        if args.artifacts is not None or args.shadow_request_output is not None:
            parser.error("shadow options require --direct-response")
        response = request_control(args.socket, observation)
    else:
        from jepa_wm.shadow_planning import ShadowPlanningRequest

        if args.artifacts is None or args.shadow_request_output is None:
            parser.error(
                "--direct-response requires --artifacts and --shadow-request-output"
            )
        if args.state is None or args.recording_root is None:
            parser.error("shadow planning requires --state and --recording-root")
        state = ControlSessionState.from_dict(json.loads(args.state.read_text()))
        validate_observation_target(
            observation,
            args.recording_root / "recordings" / state.reference_recording,
            frame_root=args.recording_root,
        )
        direct = ProposedControl.from_dict(json.loads(args.direct_response.read_text()))
        artifacts = ControlWorkerArtifacts.load(args.artifacts)
        calibration = (
            ActionResponseCalibration.load(artifacts.calibration)
            if artifacts.calibration is not None
            else None
        )
        shadow_request = ShadowPlanningRequest(
            observation,
            direct,
            artifacts.adapter,
            (
                CalibrationIdentity.from_calibration(
                    artifacts.calibration, calibration
                )
                if artifacts.calibration is not None and calibration is not None
                else None
            ),
        )
        write_json_atomic(args.shadow_request_output, shadow_request.to_dict())
        response = request_shadow_plan(
            args.socket,
            shadow_request,
        )
    payload = json.dumps(response.to_dict(), indent=2) + "\n"
    if args.direct_response is not None:
        if args.shadow_response_output is None:
            parser.error("shadow planning requires --shadow-response-output")
        write_json_atomic(args.shadow_response_output, response.to_dict())
    elif args.shadow_response_output is not None:
        parser.error("--shadow-response-output requires --direct-response")
    print(payload, end="")


if __name__ == "__main__":
    main()
