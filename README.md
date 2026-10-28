# BoViEW

BOvine VIsion for Estimating Weight

Documentation can be found at [gjburke.github.io/boview](https://gjburke.github.io/boview/pipeline/reference/)

## Setup

Set up the project with:

...your python (3.12+) environment, for example

```
python -m venv .venv
```

and then 

```
.\.venv\Scripts\Activate.ps1
```
on Windows, or
```
source ./.venv/bin/activate
```
on Mac.

And for installing the requirements:

Regular use
```
python -m pip install .
```

Development
```
python -m pip install -e ".[dev]"
```

Or instead of ".", you can also add extras ".[dev,docs]"

## Testing

You can run all the tests with

```
python -m pytest
```

and then specify the test with 

```
python -m pytest tests/<test_file>::<test_name>.py
```

## Formatting

We use ruff for formatting and linting, you can use the commands:

```
python -m ruff check .
python -m ruff format --check .
```

## Documentation

We use ProsperDocs (Continuation of version 1x MkDocs) and mkdocstrings[python] for our documentation.

To launch the documentation from the repo root:

```
python -m properdocs serve
```

And to generate a finished website, checking for build warnings:

```
python -m properdocs build --strict
```
