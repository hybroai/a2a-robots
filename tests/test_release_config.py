"""Checks that the release-please setup covers every place the version lives.

release-please bumps the version in its manifest, each package's pyproject.toml
and __init__.py (python strategy); the release-please workflow then runs
``uv lock`` on each open release PR. If these drift apart, the next release
would publish a mismatched package, and ``uv sync --locked`` would fail.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

import a2a_robots

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / ".github" / "scripts"
sys.path.insert(0, str(SCRIPTS))
import check_release_ref as check  # noqa: E402

PACKAGES = {
    ".": ("a2a-robots", "src/a2a_robots/__init__.py", "a2a-robots"),
    "packages/a2a-robots-lerobot": (
        "a2a-robots-lerobot",
        "packages/a2a-robots-lerobot/src/a2a_robots_lerobot/__init__.py",
        "a2a-robots-lerobot",
    ),
    "packages/a2a-robots-pollen": (
        "a2a-robots-pollen",
        "packages/a2a-robots-pollen/src/a2a_robots_pollen/__init__.py",
        "a2a-robots-pollen",
    ),
    "packages/a2a-robots-ros2": (
        "a2a-robots-ros2",
        "packages/a2a-robots-ros2/src/a2a_robots_ros2/__init__.py",
        "a2a-robots-ros2",
    ),
    "packages/a2a-robots-sharpa": (
        "a2a-robots-sharpa",
        "packages/a2a-robots-sharpa/src/a2a_robots_sharpa/__init__.py",
        "a2a-robots-sharpa",
    ),
}

VERSION_RE = re.compile(r'^__version__\s*=\s*["\']([^"\']+)["\']', re.M)


def _pyproject_version(package_dir: str) -> str:
    path = ROOT / package_dir / "pyproject.toml" if package_dir != "." else ROOT / "pyproject.toml"
    with open(path, "rb") as f:
        return tomllib.load(f)["project"]["version"]


def _init_version(relpath: str) -> str:
    match = VERSION_RE.search((ROOT / relpath).read_text())
    assert match, f"no __version__ in {relpath}"
    return match.group(1)


def _lock_version(lock_path: Path, package_name: str) -> str:
    with open(lock_path, "rb") as f:
        packages = tomllib.load(f)["package"]
    return next(p["version"] for p in packages if p["name"] == package_name)


def test_manifest_matches_package_versions():
    manifest = json.loads((ROOT / ".release-please-manifest.json").read_text())
    assert set(manifest) == set(PACKAGES)
    for path in PACKAGES:
        assert manifest[path] == _pyproject_version(path)


def test_all_version_locations_agree():
    assert a2a_robots.__version__ == _pyproject_version(".")
    assert a2a_robots.__version__ == _lock_version(ROOT / "uv.lock", "a2a-robots")
    for path, (pypi_name, init_path, _) in PACKAGES.items():
        version = _pyproject_version(path)
        assert _init_version(init_path) == version
        if path == ".":
            continue
        lock = ROOT / path / "uv.lock"
        assert _lock_version(lock, pypi_name) == version


def test_release_please_config_and_lock_sync():
    config = json.loads((ROOT / "release-please-config.json").read_text())
    assert config["include-component-in-tag"] is True
    assert config["separate-pull-requests"] is True
    assert set(config["packages"]) == set(PACKAGES)
    for path, (pypi_name, _, component) in PACKAGES.items():
        package = config["packages"][path]
        assert package["release-type"] == "python"
        assert package["package-name"] == pypi_name
        assert package["component"] == component
    # Root "." receives every commit unless nested package paths are excluded.
    assert set(config["packages"]["."]["exclude-paths"]) == {
        path for path in PACKAGES if path != "."
    }
    workflow = (ROOT / ".github/workflows/release-please.yml").read_text()
    assert "sync-lock:" in workflow
    assert '--label "autorelease: pending"' in workflow
    assert "uv lock\n" in workflow
    assert 'gh workflow run publish.yml --repo "$GITHUB_REPOSITORY" --ref "$tag"' in workflow


# ── publish.yml ref check ────────────────────────────────────

CHECK_SCRIPT = ROOT / ".github/scripts/check_release_ref.py"
PUBLISH_WORKFLOW = ROOT / ".github/workflows/publish.yml"
TESTPYPI_IF = "github.event_name == 'workflow_dispatch' && github.event.inputs.test_pypi == 'true'"
PYPI_IF = (
    "github.event_name == 'release' || "
    "(github.event_name == 'workflow_dispatch' && github.event.inputs.test_pypi != 'true')"
)
MISMATCH = "does not match package version"
NOT_A_TAG = "Publishing to PyPI needs a run on the"
UNKNOWN_TAG = "not a known release tag"


def _write_package_pyproject(tmp_path: Path, package_dir: str, version: str) -> None:
    if package_dir == ".":
        path = tmp_path / "pyproject.toml"
    else:
        path = tmp_path / package_dir / "pyproject.toml"
        path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f'[project]\nname = "x"\nversion = "{version}"\n')


@pytest.mark.parametrize(
    "event, ref_type, ref_name, test_pypi, package, version, code, message",
    [
        ("release", "tag", "a2a-robots-v1.2.3", None, None, "1.2.3", 0, None),
        ("release", "tag", "a2a-robots-lerobot-v1.2.3", None, None, "1.2.3", 0, None),
        ("release", "tag", "a2a-robots-v1.2.4", None, None, "1.2.3", 1, MISMATCH),
        ("release", "tag", "v1.2.3", None, None, "1.2.3", 1, UNKNOWN_TAG),
        ("workflow_dispatch", "tag", "a2a-robots-v1.2.3", "false", None, "1.2.3", 0, None),
        ("workflow_dispatch", "tag", "a2a-robots-v1.2.3", None, None, "1.2.3", 0, None),
        ("workflow_dispatch", "branch", "main", "false", "a2a-robots", "1.2.3", 1, NOT_A_TAG),
        ("workflow_dispatch", "branch", "main", None, "a2a-robots", "1.2.3", 1, NOT_A_TAG),
        ("workflow_dispatch", "branch", "feature", "true", "a2a-robots", "1.2.3", 0, None),
        (
            "workflow_dispatch",
            "branch",
            "feature",
            "true",
            "a2a-robots-pollen",
            "1.2.3",
            0,
            None,
        ),
        ("workflow_dispatch", "tag", "a2a-robots-v9.9.9", "true", None, "1.2.3", 1, MISMATCH),
        ("workflow_dispatch", "branch", "main", " true", "a2a-robots", "1.2.3", 1, NOT_A_TAG),
        ("workflow_dispatch", "branch", "main", "TRUE", "a2a-robots", "1.2.3", 0, None),
        ("push", "branch", "main", None, None, "1.2.3", 1, "unexpected event"),
    ],
)
def test_release_ref_check(
    tmp_path, event, ref_type, ref_name, test_pypi, package, version, code, message
):
    parsed = check.parse_component_tag(ref_name) if ref_type == "tag" else None
    if parsed is not None:
        package_dir = check.PACKAGES[parsed[0]][0]
    else:
        package_dir = check.PACKAGES[(package or "a2a-robots")][0]
    _write_package_pyproject(tmp_path, package_dir, version)

    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("GITHUB_") and key not in {"TEST_PYPI", "PACKAGE"}
    }
    env.update(GITHUB_EVENT_NAME=event, GITHUB_REF_TYPE=ref_type, GITHUB_REF_NAME=ref_name)
    if test_pypi is not None:
        env["TEST_PYPI"] = test_pypi
    if package is not None:
        env["PACKAGE"] = package

    result = subprocess.run(
        [sys.executable, str(CHECK_SCRIPT)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == code, result.stdout + result.stderr
    if message:
        assert "::error::" in result.stdout
        assert message in result.stdout


def _job_block(workflow: str, job: str) -> str:
    match = re.search(rf"^  {re.escape(job)}:\n(.*?)(?=^  \S|\Z)", workflow, re.M | re.S)
    assert match, f"no {job} job"
    return match.group(0)


def test_publish_workflow_runs_the_ref_check():
    workflow = PUBLISH_WORKFLOW.read_text()
    check_job = _job_block(workflow, "check-ref")
    assert "python .github/scripts/check_release_ref.py" in check_job
    assert not re.search(r"^\s*(- )?if:", check_job, re.M)
    assert "TEST_PYPI: ${{ github.event.inputs.test_pypi }}" in check_job
    assert "needs: check-ref" in _job_block(workflow, "test")
    assert f"if: {TESTPYPI_IF}\n" in _job_block(workflow, "publish-to-testpypi")
    assert f"if: {PYPI_IF}\n" in _job_block(workflow, "publish-to-pypi")
    ci = (ROOT / ".github/workflows/ci.yml").read_text()
    assert "workflow_call:" in ci


def test_component_tag_prefers_longest_match():
    assert check.parse_component_tag("a2a-robots-lerobot-v0.1.0") == (
        "a2a-robots-lerobot",
        "0.1.0",
    )
    assert check.parse_component_tag("a2a-robots-v0.1.0") == ("a2a-robots", "0.1.0")
    assert check.parse_component_tag("v0.1.0") is None
