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

## Style Enforcement

Before adjusting whitespace or other stylistic details, run the test suite to
get a baseline:

```bash
pytest -q --disable-warnings --maxfail=1
```

After applying formatting changes—such as removing blank lines directly under
decorators, collapsing any triple blank lines to two, and ensuring files end
with a single newline—run the same command again to confirm all tests still
pass. This ensures cosmetic refactors never introduce regressions.
