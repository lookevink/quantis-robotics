"""Send one versioned observation to the resident Unix-socket worker."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import socket

from jepa_wm.control_protocol import ControlObservation, ProposedControl


def request_control(socket_path: Path, observation: ControlObservation) -> ProposedControl:
    request = json.dumps(observation.to_dict(), separators=(",", ":")) + "\n"
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
    return ProposedControl.from_dict(json.loads(lines[0]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--socket", type=Path, required=True)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--response", type=Path)
    args = parser.parse_args()
    observation = ControlObservation.from_dict(json.loads(args.request.read_text()))
    response = request_control(args.socket, observation)
    payload = json.dumps(response.to_dict(), indent=2) + "\n"
    if args.response is None:
        print(payload, end="")
        return
    temporary = args.response.with_suffix(args.response.suffix + ".tmp")
    temporary.write_text(payload)
    temporary.replace(args.response)
    print(payload, end="")


if __name__ == "__main__":
    main()
