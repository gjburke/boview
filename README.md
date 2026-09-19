# BoViEW

BOvine VIsion for Estimating Weight

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

```
python -m pip install -e ".[dev]"
```

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

