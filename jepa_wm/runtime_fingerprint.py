"""Conservative authentication of repository-owned executable behavior."""

from __future__ import annotations

from hashlib import sha256
import os
from pathlib import Path


RUNTIME_SOURCE_SUFFIXES = frozenset((".py", ".sh"))
RUNTIME_SOURCE_EXCLUDED_DIRECTORIES = frozenset(
    ("data", "outputs", "tests", "__pycache__")
)


def runtime_source_files(repository: Path) -> tuple[str, ...]:
    """Discover every repository-owned Python and shell runtime source file."""

    root = repository.resolve()
    files = []
    for directory, directory_names, filenames in os.walk(root):
        directory_names[:] = [
            name
            for name in directory_names
            if name not in RUNTIME_SOURCE_EXCLUDED_DIRECTORIES
            and not name.startswith(".")
        ]
        parent = Path(directory)
        files.extend(
            (parent / filename).relative_to(root).as_posix()
            for filename in filenames
            if (parent / filename).suffix in RUNTIME_SOURCE_SUFFIXES
        )
    result = tuple(sorted(set(files)))
    if not result:
        raise ValueError("repository has no discoverable runtime source files")
    return result


def runtime_source_fingerprint(repository: Path | None = None) -> str:
    """Hash names and contents so additions, removals, and edits all invalidate."""

    root = (
        repository.resolve()
        if repository is not None
        else Path(__file__).resolve().parents[1]
    )
    digest = sha256()
    for relative in runtime_source_files(root):
        encoded = relative.encode()
        contents = (root / relative).read_bytes()
        digest.update(len(encoded).to_bytes(4, "big"))
        digest.update(encoded)
        digest.update(len(contents).to_bytes(8, "big"))
        digest.update(contents)
    return digest.hexdigest()
