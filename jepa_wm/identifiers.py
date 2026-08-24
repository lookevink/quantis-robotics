"""Dependency-light validation for persisted Quantis artifact identifiers."""

from __future__ import annotations

import re


_SAFE_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def validate_safe_identifier(identifier: str) -> None:
    if not _SAFE_IDENTIFIER.fullmatch(identifier):
        raise ValueError(
            "identifier must contain only letters, numbers, dot, dash, or underscore"
        )
