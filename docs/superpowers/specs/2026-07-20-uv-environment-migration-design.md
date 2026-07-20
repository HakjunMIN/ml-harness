# uv Environment Migration Design

## Goal

Make uv the single reproducible development environment for this repository.
Dependencies remain declared in `pyproject.toml`; `uv.lock` pins the resolved
environment and is committed.

## Scope

- Generate and commit `uv.lock`.
- Keep the existing runtime, `dev`, and `dashboard` dependency groups intact.
- Change documented setup, test, CLI, and dashboard commands to `uv sync` and
  `uv run`.
- Run harness Python entry points through `uv run` so they use the locked
  project environment.
- Preserve `.venv/` as an ignored local artifact.

## Non-goals

- No dependency upgrades beyond resolutions required to create the lockfile.
- No change to the package build backend or supported Python range.
- No separate virtual-environment manager, requirements file, or cache checked
  into the repository.

## Workflow

Developers create the project environment with:

```bash
uv sync --all-extras
```

They run commands through the locked environment:

```bash
uv run pytest -q
uv run python -m power_forecasting.cli all --output artifacts/demo --days 60 --plants 3 --seed 42
```

The checked-in lockfile is authoritative for installed package versions.
Harness scripts invoke `uv run python` rather than an ambient `python3`.

## Validation

Verify that `uv lock --locked` accepts the committed lockfile, then run the
smallest existing test coverage affected by changed executable paths and
documentation commands.
