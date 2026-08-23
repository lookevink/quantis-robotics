"""Unix-socket server that keeps JEPA-WM control inference resident on the GPU."""

from __future__ import annotations

import argparse
from io import StringIO
from pathlib import Path
import socketserver

from jepa_wm.control_worker import FrozenProposalPredictor, serve_jsonl


class _ControlRequestHandler(socketserver.StreamRequestHandler):
    def handle(self) -> None:
        output = StringIO()
        serve_jsonl(
            StringIO(self.rfile.read().decode("utf-8")),
            output,
            self.server.predictor,
        )
        self.wfile.write(output.getvalue().encode("utf-8"))


class ControlUnixServer(socketserver.UnixStreamServer):
    def __init__(self, socket_path: Path, predictor) -> None:
        self.socket_path = socket_path.resolve()
        self.socket_path.parent.mkdir(parents=True, exist_ok=True)
        self.socket_path.unlink(missing_ok=True)
        self.predictor = predictor
        super().__init__(str(self.socket_path), _ControlRequestHandler)

    def server_close(self) -> None:
        try:
            super().server_close()
        finally:
            self.socket_path.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--proposal", type=Path, required=True)
    parser.add_argument("--socket", type=Path, required=True)
    parser.add_argument("--frame-root", type=Path)
    args = parser.parse_args()
    predictor = FrozenProposalPredictor(
        args.source,
        args.checkpoint,
        args.proposal,
        frame_root=args.frame_root,
    )
    with ControlUnixServer(args.socket, predictor) as server:
        server.serve_forever()


if __name__ == "__main__":
    main()
