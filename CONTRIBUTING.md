# Contributing

## Release Process

**(For maintainers only)**

Releases are automated with [release-please](https://github.com/googleapis/release-please).
Do not bump versions or add CHANGELOG entries by hand.

Prerequisite: "Allow GitHub Actions to create and approve pull requests" must
stay enabled in the repository's Actions settings, or release-please cannot
open its PRs.

Each installable package versions independently:

| Package | Component / tag prefix | Directory |
| --- | --- | --- |
| `a2a-robots` | `a2a-robots` | `.` |
| `a2a-robots-lerobot` | `a2a-robots-lerobot` | `packages/a2a-robots-lerobot` |
| `a2a-robots-pollen` | `a2a-robots-pollen` | `packages/a2a-robots-pollen` |
| `a2a-robots-ros2` | `a2a-robots-ros2` | `packages/a2a-robots-ros2` |
| `a2a-robots-sharpa` | `a2a-robots-sharpa` | `packages/a2a-robots-sharpa` |

The root package configures `exclude-paths` for every `packages/*` directory so
backend-only commits do not also bump `a2a-robots`.

1. **Merge changes into `main` with Conventional Commit messages.**
   release-please attributes commits to packages by the paths they touch.
   Prefer scoped subjects such as `feat(lerobot): ...` or `fix(pollen): ...`.
   - `fix:`, `perf:` and `revert:` bump the patch version. `feat:` bumps it
     too while the version is below 1.0.0, and the minor version from 1.0.0 on.
   - `feat!:` or a `BREAKING CHANGE:` footer bumps the minor version below
     1.0.0, and the major version from 1.0.0 on.
   - `docs:`, `chore:`, `ci:`, `test:` and the other types do not start a release.
2. **Review the release PR.** After each push to `main`, release-please opens
   or updates a `chore(main): release <package> X.Y.Z` PR per package that has
   releasable commits. It bumps the version in that package's `pyproject.toml`,
   `__init__.py`, and `.release-please-manifest.json`, and updates its
   `CHANGELOG.md`. The workflow then pushes a
   `chore: sync uv.lock with the release version` commit when locks need it.
   Edit the PR if the changelog needs wording changes.

   CI does not run on these PRs, because GitHub starts no workflows for PRs
   opened by `GITHUB_TOKEN`. Merge only once each touched `uv.lock` shows the
   new version.
3. **Merge the release PR.** release-please creates the
   `{component}-vX.Y.Z` tag and the GitHub Release, then dispatches
   `publish.yml` on that tag. The workflow runs CI, checks that the tag matches
   the package version, builds that package, and publishes it to PyPI after a
   maintainer approves the `pypi` environment deployment.

If publishing fails, fix the cause and re-run `publish.yml` from the Actions
tab on the `{component}-vX.Y.Z` tag. A hand run publishes to PyPI only from a
matching tag; from a branch, run it with `test_pypi=true` and choose the
package.

### Versioning

We follow [Semantic Versioning](https://semver.org/). Until 1.0.0, versions
shift down one level: breaking changes bump MINOR, and new features and fixes
bump PATCH.

### PyPI Trusted Publishing

Each package name is registered as a Trusted Publisher for workflow
`publish.yml` with environment `pypi` (and `testpypi` on TestPyPI). Create the
GitHub Environments `pypi` and `testpypi` with required reviewers.
