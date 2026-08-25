"""Dependency-light identity for the insertion control-resolution benchmark."""

from __future__ import annotations

import argparse
from enum import Enum


class ControlResolutionLoad(str, Enum):
    ATTACHED = "attached"
    UNLOADED = "unloaded"

    @property
    def plug_attached(self) -> bool:
        return self is ControlResolutionLoad.ATTACHED


CONTROL_RESOLUTION_CONTEXTS = (43, 74, 106)
CONTROL_RESOLUTION_LOADS = tuple(ControlResolutionLoad)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Print the canonical insertion resolution benchmark profile."
    )
    parser.add_argument("field", choices=("contexts", "loads", "roster", "load"))
    parser.add_argument(
        "--load",
        choices=tuple(load.value for load in ControlResolutionLoad),
    )
    args = parser.parse_args()
    if args.field == "contexts":
        print("\n".join(str(context) for context in CONTROL_RESOLUTION_CONTEXTS))
    elif args.field == "loads":
        print("\n".join(load.value for load in CONTROL_RESOLUTION_LOADS))
    elif args.field == "roster":
        print(
            "\n".join(
                f"{context}\t{load.value}"
                for load in CONTROL_RESOLUTION_LOADS
                for context in CONTROL_RESOLUTION_CONTEXTS
            )
        )
    elif args.load is None:
        parser.error("load field requires --load")
    else:
        print(ControlResolutionLoad(args.load).value)


if __name__ == "__main__":
    main()
