# AGENTS.md

This file provides context and instructions for AI agents working on the Mars repository.

## Repository Overview

Mars (Mass Accuracy Recalibration System) is a tool for calibrating DIA mass spectrometry data from the Thermo Stellar instrument. It uses XGBoost to learn m/z corrections from spectral library matches.

## Continuous Integration (CI/CD)

The repository uses GitHub Actions for CI/CD, defined in `.github/workflows/`.

### Workflows

1.  **Tests (`tests.yml`)**
    *   **Triggers:** Push to `main`, Pull Requests to `main`.
    *   **Actions:**
        *   Sets up Python 3.10, 3.11, 3.12.
        *   Installs dependencies with `pip install -e ".[dev]"`.
        *   Runs tests using `pytest tests/ -v --tb=short`.
        *   Runs linting with `ruff check mars/`.

2.  **Publish to PyPI (`publish.yml`)**
    *   **Triggers:** Release published.
    *   **Actions:**
        *   Builds the package (`python -m build`).
        *   Publishes to PyPI using Trusted Publishing (OIDC).
        *   Requires the tag (e.g., `v0.1.0`) to match the release.

## Common Development Tasks

*   **Install for dev:** `pip install -e ".[dev]"`
*   **Run tests:** `pytest tests/`
*   **Lint:** `ruff check .`
*   **Build package:** `python -m build`

## Release Notes

When making bug fixes or improvements during development:

*   **Update the current release notes:** Add any fixes or changes to the current version's release notes file in `release-notes/` (e.g., `RELEASE_NOTES_v0.1.3.md`).
*   **Be specific:** Document what was fixed and why.
*   **Group related changes:** Use appropriate sections (Bug Fixes, Changes, New Features, etc.).

## Style Guidelines

*   **No Emojis:** Do not use emojis in any output, documentation, source code comments, or Jupyter notebooks. Keep all text professional and plain.
