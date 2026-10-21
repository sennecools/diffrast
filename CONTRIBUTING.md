# Contributing

Thanks for your interest in diffrast. Bug reports, test cases from real pipelines, and benchmark
results on hardware we do not have are all especially welcome.

## Clean-room policy

diffrast is an independent, MIT-licensed implementation. To keep it that way:

- **Do not read, copy or translate nvdiffrast source code** (or any other code under a
  non-permissive license) while working on diffrast.
- Public documentation and papers may be used for API names and documented behaviour.
- Tests are written from first principles: analytic scenes, finite differences and
  `torch.autograd.gradcheck`, not ported from another project's test suite.

Pull requests that appear to be derived from restricted code will be closed.

## Setup

```bash
git clone https://github.com/sennecools/diffrast
cd diffrast
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip  # --group needs pip 25.1+
pip install torch                    # or the CUDA / ROCm build for your GPU
pip install -e . --group dev
pre-commit install
```

## Tests

```bash
pytest
```

Tests marked `gpu` are skipped when no CUDA or ROCm device is available. If you change a kernel,
please run the suite on a GPU and mention the device in your pull request.

New ops and backends should be tested against the reference backend, and gradients checked with
`torch.autograd.gradcheck` in float64 on small scenes.

## Style

Formatting and linting are handled by [ruff](https://docs.astral.sh/ruff/):

```bash
ruff check .
ruff format .
```

Public functions have docstrings that state tensor shapes and dtypes.

## Pull requests

- Keep each pull request focused on one change.
- Add or update tests for any behaviour change.
- Add an entry under "Unreleased" in [CHANGELOG.md](CHANGELOG.md) for user-visible changes.
