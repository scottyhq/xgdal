# Claude Instructions

## Python Environment
Always use `pixi run python ...` when running Python commands. Do NOT create virtual environments (no `python -m venv`, `conda create`, `uv venv`, etc.). All Python execution should go through pixi.

## Common Commands

```bash
# Run tests
pixi run test

# Lint code
pixi run lint        # ruff check src tests

# Format code
pixi run format      # ruff format src tests
```

## Code Style

This project uses [ruff](https://docs.astral.sh/ruff/) for linting and formatting.

- Run `pixi run lint` to check for issues; fix violations before committing
- Run `pixi run format` to auto-format code
- Do not add type annotations, docstrings, or comments to code you did not change
- Keep changes minimal and focused — avoid refactoring unrelated code

## Project Structure

```
src/xgdal/
  __init__.py   # Public exports
  backend.py    # Xarray BackendEntrypoint implementation
  accessor.py   # Xarray accessor
  env.py        # GDAL environment management
  vrt.py        # VRT XML construction
tests/
  test_backend.py
```

- Source lives under `src/xgdal/` (src layout)
- The package registers itself as an Xarray backend via the `xarray.backends` entry point in `pyproject.toml`
