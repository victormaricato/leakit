# leakit

**Continuation-free membership inference for closed language models.**

`leakit` tells you whether a document was likely in a model's training set using
nothing but its sampling API. No logits, no log-probabilities, and -- unlike
prior sampling attacks such as SaMIA -- no need to know the document's true
continuation. You give it the *opening* of a document; it samples several
continuations and measures how much they agree with each other. Training
documents pull the model's continuation distribution toward the memorised text,
so the samples concentrate; novel documents leave the distribution diffuse.

This is the reference implementation of the *self-concentration* attack from the
paper *"Leak It: Continuation-Free Membership Inference on Closed Language Models
via Sample Self-Concentration."*

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
service you are probing -- the key maps to whatever provider `--base-url` points
at -- then run it.

```bash
export LEAKIT_API_KEY="sk-..."        # or OPENAI_API_KEY

# OpenAI
leakit --model gpt-4o-mini suspect.txt

# Anything OpenAI-compatible (OpenRouter, Anthropic compat route, vLLM, Together, local server)
leakit --model anthropic/claude-3.5-sonnet \
       --base-url https://openrouter.ai/api/v1 \
       --api-key-env OPENROUTER_API_KEY \
       -n 32 suspect.txt

# Compare a candidate against known non-member documents (relative percentile)
leakit --model gpt-4o-mini --calibrate clean/*.txt suspect.txt

# Pipe text in, get JSON out
cat article.txt | leakit --model gpt-4o-mini --json
```

Output:

```
document     score  samples
-----------------------------
suspect.txt  0.4213   32/32
```

A higher score means the sampled continuations agree more, which correlates with
membership. The absolute scale is model-dependent, so interpret scores
*relatively*: score several documents together, or use `--calibrate` with a set
of documents you know were **not** in training to get a percentile.

### Key options

| Flag | Meaning | Default |
|------|---------|---------|
| `--model` | model id passed to the API | required |
| `--base-url` | OpenAI-compatible endpoint | OpenAI |
| `--api-key-env` | env var holding the key | `LEAKIT_API_KEY`, then `OPENAI_API_KEY` |
| `-n, --samples` | continuations per document | 16 |
| `--max-tokens` | tokens per continuation | 64 |
| `--temperature` | sampling temperature | 1.0 |
| `--prefix-chars` | chars of each doc used as the prefix (0 = whole doc) | 256 |
| `--statistic` | `word-jaccard` (parameter-free) or `kgram` | `word-jaccard` |
| `--mode` | `chat` (closed APIs) or `completion` (base models) | `chat` |
| `--calibrate` | non-member baseline file(s) for a percentile | off |
| `--json` | machine-readable output | off |

For base/text-completion models (e.g. self-hosted Pythia/Llama base), use
`--mode completion` to sample the raw continuation distribution. For chat/instruct
models, the default `chat` mode asks the model to continue the passage verbatim.

## Python API

```python
from leakit import LeakIt

scorer = LeakIt(model="gpt-4o-mini", n_samples=32)   # reads LEAKIT_API_KEY/OPENAI_API_KEY
result = scorer.score(open("suspect.txt").read())
print(result.score, result.n_returned)
```

The raw statistics are exposed too:

```python
from leakit import self_concentration_word_jaccard
self_concentration_word_jaccard(["a b c", "a b c", "x y z"])
```

## Responsible use

`leakit` is a privacy-auditing and red-teaming tool: use it to test models you
own or are authorised to assess. The self-concentration signal is a statistical
indicator, not proof of membership; calibrate before drawing conclusions.

## License

MIT.
