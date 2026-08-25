"""Resident cache for immutable reset-trial source evidence."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Generic, TypeVar

from sim.control_session import ControlSession


Evidence = TypeVar("Evidence")


@dataclass
class ResidentTrialSourceCache(Generic[Evidence]):
    _sources: dict[tuple[Path, str], Evidence] = field(default_factory=dict)

    @staticmethod
    def _key(source_session: ControlSession) -> tuple[Path, str]:
        return source_session.path.parent.resolve(), source_session.session_id

    def prepare(
        self,
        source_session: ControlSession,
        loader: Callable[[], Evidence],
    ) -> Evidence:
        evidence = loader()
        self._sources[self._key(source_session)] = evidence
        return evidence

    def consume(
        self,
        source_session: ControlSession,
        loader: Callable[[], Evidence],
    ) -> Evidence:
        evidence = self._sources.pop(self._key(source_session), None)
        return loader() if evidence is None else evidence
