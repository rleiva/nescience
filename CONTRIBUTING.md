# Contributing

Thanks for helping improve **nescience**!

## Environment

```bash
python -m venv .venv && source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -e .[dev]
pre-commit install
```

## Testing

```bash
pytest -q --cov=nescience --cov-report=term-missing
```

## Releasing (maintainers)

- Bump version in `pyproject.toml`
- `python -m build`
- Trusted Publishing from CI (GitHub Actions).
