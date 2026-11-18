"""Check that a publish.yml run is on the release tag before anything is built.

publish.yml publishes to PyPI on a ``release`` event, or on a hand run
(``workflow_dispatch``) unless ``test_pypi`` is true. Such a run must be on the
``{component}-v{version}`` tag for the version in that package's pyproject.toml.
A TestPyPI rehearsal may run from a branch; on a tag, the tag must still match.

The publish job conditions in publish.yml decide what a run publishes, and this
script mirrors them. tests/test_release_config.py pins both; change them
together.
"""

from __future__ import annotations

import os
import sys
import tomllib
from pathlib import Path

# Longest component first so a2a-robots-lerobot wins over a2a-robots.
PACKAGES: dict[str, tuple[str, str]] = {
    "a2a-robots-lerobot": ("packages/a2a-robots-lerobot", "a2a-robots-lerobot"),
    "a2a-robots-pollen": ("packages/a2a-robots-pollen", "a2a-robots-pollen"),
    "a2a-robots-ros2": ("packages/a2a-robots-ros2", "a2a-robots-ros2"),
    "a2a-robots-sharpa": ("packages/a2a-robots-sharpa", "a2a-robots-sharpa"),
    "a2a-robots": (".", "a2a-robots"),
}

_COMPONENTS = tuple(sorted(PACKAGES, key=len, reverse=True))


def _error(message: str) -> int:
    print(f"::error::{message}")
    return 1


def parse_component_tag(ref_name: str) -> tuple[str, str] | None:
    """Return (component, version) for ``{component}-v{version}``, else None."""
    for component in _COMPONENTS:
        prefix = f"{component}-v"
        if ref_name.startswith(prefix):
            version = ref_name[len(prefix) :]
            if version and version[0].isdigit():
                return component, version
    return None


def resolve_package(
    *,
    ref_type: str,
    ref_name: str,
    package_input: str,
) -> tuple[str, str, str, Path]:
    """Return (component, package_dir, package_name, pyproject_path)."""
    if ref_type == "tag":
        parsed = parse_component_tag(ref_name)
        if parsed is None:
            raise ValueError(
                f"Tag {ref_name} is not a known release tag "
                f"(expected one of: {', '.join(c + '-v*' for c in _COMPONENTS)})."
            )
        component, _ = parsed
    else:
        component = package_input.strip() or "a2a-robots"
        if component not in PACKAGES:
            raise ValueError(
                f"Unknown package {component!r}; expected one of: {', '.join(_COMPONENTS)}."
            )

    package_dir, package_name = PACKAGES[component]
    pyproject = Path(package_dir) / "pyproject.toml"
    return component, package_dir, package_name, pyproject


def _write_output(name: str, value: str) -> None:
    output_file = os.environ.get("GITHUB_OUTPUT")
    if not output_file:
        return
    with open(output_file, "a", encoding="utf-8") as f:
        f.write(f"{name}={value}\n")


def main() -> int:
    event = os.environ.get("GITHUB_EVENT_NAME", "")
    ref_type = os.environ.get("GITHUB_REF_TYPE", "")
    ref_name = os.environ.get("GITHUB_REF_NAME", "")
    # Actions compares strings in ``if:`` ignoring case but not whitespace, so
    # " true" still publishes to PyPI. No strip(): fail closed.
    test_pypi = os.environ.get("TEST_PYPI", "").lower() == "true"
    package_input = os.environ.get("PACKAGE", "")

    try:
        component, package_dir, package_name, pyproject = resolve_package(
            ref_type=ref_type,
            ref_name=ref_name,
            package_input=package_input,
        )
    except ValueError as exc:
        return _error(str(exc))

    with open(pyproject, "rb") as f:
        version = tomllib.load(f)["project"]["version"]
    tag = f"{component}-v{version}"

    _write_output("component", component)
    _write_output("package_dir", package_dir)
    _write_output("package_name", package_name)
    _write_output("version", version)

    if event == "release" or (event == "workflow_dispatch" and not test_pypi):
        if ref_type != "tag":
            return _error(
                f"Publishing to PyPI needs a run on the {tag} tag, not {ref_type} "
                f"{ref_name}; run from a branch only with test_pypi=true."
            )
    elif event == "workflow_dispatch":
        if ref_type != "tag":
            print(f"TestPyPI rehearsal from {ref_type} {ref_name} for {package_name}.")
            return 0
    else:
        return _error(f"publish.yml ran on an unexpected event: {event or '(none)'}.")

    if ref_name != tag:
        return _error(
            f"Release tag {ref_name} does not match package version {tag}. Release "
            "through the release-please PR so the version is bumped first."
        )
    print(f"Ref {ref_name} matches package version {tag} ({package_dir}).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
