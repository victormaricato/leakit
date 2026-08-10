# leakit

[![PyPI](https://img.shields.io/pypi/v/leakit.svg)](https://pypi.org/project/leakit/)
[![Python](https://img.shields.io/pypi/pyversions/leakit.svg)](https://pypi.org/project/leakit/)
[![CI](https://github.com/victormaricato/leakit/actions/workflows/ci.yml/badge.svg)](https://github.com/victormaricato/leakit/actions/workflows/ci.yml)
[![arXiv](https://img.shields.io/badge/arXiv-2608.00144-b31b1b.svg)](https://arxiv.org/abs/2608.00144)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Black-box training-data extraction auditing for language models.**

`leakit` asks whether a model reproduces a *specific string* from your document
verbatim, using nothing but its sampling API. No logits, no log-probabilities, no
weights. You point it at a document you own and a string you care about (an email
address, a phone number, an API key, a paragraph); it conditions the model on the
text immediately preceding that string, samples continuations, and checks whether
any reproduces it exactly.

Reference implementation for
[arXiv:2608.00144](https://arxiv.org/abs/2608.00144), *"Leak It: A Probabilistic
Approach to Training-Data Extraction from Black-Box Language Models."*

## Why the control matters

A raw reproduction rate overstates the harm. A model will emit a globally common
string, like a project's contact mailbox, from *any* context. So `leakit` treats a
leak as attributable to your document only when both hold:

- the string is reproduced from **its own preceding context**, and
- it is **not** reproduced from a **mismatched** context (`--control`).

That conjunction is evaluated per document, not as a difference of two population
rates, which is what licenses a claim about an individual document. Without
`--control`, `leakit` reports attribution as `?` rather than guessing.

## Install

```bash
curl -fsSL https://raw.githubusercontent.com/victormaricato/leakit/main/install.sh | bash
```

or, directly, with any of:

```bash
uv tool install leakit       # recommended
pipx install leakit
pip install leakit
```

## Use

`leakit` talks to any **OpenAI-compatible** endpoint. Set the API key for the
service you are probing, then run it.

```bash
export LEAKIT_API_KEY="sk-..."        # or OPENAI_API_KEY

# Does the model reproduce this exact string from its own context, and not
# from an unrelated one?
leakit --model gpt-4o-mini \
       --target "alice@example.org" \
       --control unrelated.txt  mydoc.txt

# Find candidate identifiers automatically, then probe each
leakit --model gpt-4o-mini --detect email --control unrelated.txt mydoc.txt

# Any OpenAI-compatible endpoint (OpenRouter, vLLM, Together, local server)
leakit --model anthropic/claude-3.5-sonnet \
       --base-url https://openrouter.ai/api/v1 \
       --api-key-env OPENROUTER_API_KEY \
       --detect email --control unrelated.txt -n 32  mydoc.txt
```

Output:

```
document   target         repro  control  attributable     hits
---------------------------------------------------------------
mydoc.txt  a***@e***.org    yes       no           yes    4/32
```

`attributable = yes` means the model reproduced **that document's** string and
does not emit it generically. Identifiers are **masked by default** so that
reporting a leak does not republish it; pass `--show-identifiers` to opt out.

### Key options

| Flag | Meaning | Default |
|------|---------|---------|
| `--model` | model id passed to the API | required |
| `--target` | exact string to test (repeatable) | — |
| `--detect` | auto-detect targets: `email` or `phone` | — |
| `--control` | unrelated document giving the mismatched-prefix control | off |
| `--show-identifiers` | print identifiers unmasked | off (masked) |
| `--base-url` | OpenAI-compatible endpoint | OpenAI |
| `--api-key-env` | env var holding the key | `LEAKIT_API_KEY`, then `OPENAI_API_KEY` |
| `-n, --samples` | continuations per probe | 16 |
| `--max-tokens` | tokens per continuation | 64 |
| `--temperature` / `--top-p` | decoding | 1.0 / 1.0 |
| `--prefix-chars` | chars of context before the target | 256 |
| `--mode` | `chat` (closed APIs) or `completion` (base models) | `chat` |
| `--json` | machine-readable output | off |

For base/text-completion models (self-hosted Pythia/Llama base), use
`--mode completion`. For chat/instruct models the default asks the model to
continue the passage verbatim.

The paper's regime findings transfer directly to how you configure this:
temperature and nucleus sampling matter little, a 16-token prefix already
suffices, and the samples you need scale inversely with prefix length, so a
longer `--prefix-chars` lets you cut `-n`.

## Python API

```python
from leakit import Extractor

ex = Extractor(model="gpt-4o-mini", n_samples=32)
r = ex.probe(document, target="alice@example.org", control_context=unrelated)
r.attributable        # True / False / None (None = no control supplied)
r.n_hits              # how many sampled continuations reproduced it
```

## Aggregate membership scoring (diagnostic only)

`leakit` also computes the aggregate sampling statistics studied in the paper
(*self-concentration*: how much the sampled continuations agree with each other),
via the default mode with no `--target`/`--detect`:

```bash
leakit --model gpt-4o-mini suspect.txt
```

**Read the caveat before using this as evidence.** The paper's own finding is
that these aggregate statistics do **not** beat a model-free *blind* baseline: a
bag-of-words classifier that never queries the model reaches AUC 0.97 on WikiMIA,
and on an IID Pile split the incremental AUC from sampling has a 95% CI that
includes zero. Aggregate membership scores are kept here for reproducibility and
as a diagnostic, **not** as a membership test. If you want evidence about a
document, use the extraction probe above, which a blind attack cannot fake.

## Responsible use

`leakit` is a privacy-auditing and red-teaming tool: use it on models and
documents you own or are authorised to assess. Identifiers are masked in output by
default. Do not use it to harvest identifiers from text you do not control.

## Citing

```bibtex
@misc{maricato2026leakit,
  title         = {Leak It: A Probabilistic Approach to Training-Data Extraction
                   from Black-Box Language Models},
  author        = {Victor Maricato},
  year          = {2026},
  eprint        = {2608.00144},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  url           = {https://arxiv.org/abs/2608.00144}
}
```

## Development

Uses [uv](https://docs.astral.sh/uv/). Lint with Ruff, test with pytest.

```bash
uv sync                       # install deps into .venv from uv.lock
uv run ruff check             # lint
uv run ruff format            # format
uv run pytest                 # unit tests (integration excluded by default)
uv run pytest -m integration  # hermetic HTTP integration tests
```

CI (GitHub Actions) runs lint and the full test matrix (Python 3.9 / 3.11 /
3.13) on every push and PR. Releases publish to PyPI via trusted publishing on a
`v*` tag or a published GitHub Release.

## License

MIT.
