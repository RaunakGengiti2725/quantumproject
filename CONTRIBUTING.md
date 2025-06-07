# Contributing

This project uses Black, isort, and flake8 for code style.

## Setup

```bash
pip install black isort flake8
```

## Formatting

Run the formatters from the repository root:

```bash
isort . --profile black
black . --line-length 88
```

Check for lint errors with:

```bash
flake8
```

All code committed to the repository should pass these commands as well as the unit tests:

```bash
pytest -q
```
