"""Command-line interface for leakit."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import __version__, _stats
from .core import LeakIt, ScoreResult, percentile_of

_EPILOG = """\
examples:
  # score a file against an OpenAI model (reads LEAKIT_API_KEY or OPENAI_API_KEY)
  leakit --model gpt-4o-mini suspect.txt

  # probe a model served behind any OpenAI-compatible endpoint
  leakit --model anthropic/claude-3.5-sonnet \\
         --base-url https://openrouter.ai/api/v1 \\
         --api-key-env OPENROUTER_API_KEY \\
         -n 32 suspect.txt

  # compare a candidate against a baseline of known non-member documents
  leakit --model gpt-4o-mini --calibrate known_clean/*.txt suspect.txt

  # pipe text in
  cat article.txt | leakit --model gpt-4o-mini

A higher self-concentration score means the model's continuations agree more,
which the paper shows correlates with training-set membership. The absolute
scale is model-dependent; use --calibrate (or score several documents together)
to interpret a score relatively.
"""


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="leakit",
        description="Continuation-free membership inference on closed language models.",
        epilog=_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("documents", nargs="*", help="document file(s) to score; omit to read stdin")
    p.add_argument("--text", action="append", default=[], metavar="STR",
                   help="inline document text (repeatable)")
    p.add_argument("--model", required=True, help="model id passed to the API")
    p.add_argument("--base-url", default=None,
                   help="OpenAI-compatible base URL (default: OpenAI)")
    p.add_argument("--api-key-env", default=None, metavar="VAR",
                   help="env var holding the API key (default: LEAKIT_API_KEY then OPENAI_API_KEY)")
    p.add_argument("-n", "--samples", type=int, default=16,
                   help="number of sampled continuations per document (default: 16)")
    p.add_argument("--max-tokens", type=int, default=64,
                   help="tokens generated per continuation (default: 64)")
    p.add_argument("--temperature", type=float, default=1.0, help="sampling temperature (default: 1.0)")
    p.add_argument("--top-p", type=float, default=1.0, help="nucleus sampling top-p (default: 1.0)")
    p.add_argument("--prefix-chars", type=int, default=256,
                   help="chars of each document used as the conditioning prefix; 0 = whole document (default: 256)")
    p.add_argument("--statistic", choices=sorted(_stats.STATISTICS), default="word-jaccard",
                   help="self-concentration statistic (default: word-jaccard)")
    p.add_argument("--k", type=int, default=5, help="k for the kgram statistic (default: 5)")
    p.add_argument("--mode", choices=("chat", "completion"), default="chat",
                   help="API surface: chat for instruct/closed APIs (default), completion for base models")
    p.add_argument("--concurrency", type=int, default=8, help="parallel requests (default: 8)")
    p.add_argument("--n-per-request", type=int, default=1,
                   help="continuations per API call via the provider's n param (default: 1)")
    p.add_argument("--calibrate", default=None, metavar="GLOB",
                   help="glob (quote it) or comma-separated path(s) of known NON-member documents; "
                        "report each candidate's percentile vs this baseline")
    p.add_argument("--json", action="store_true", help="emit JSON instead of a table")
    p.add_argument("--show-completions", action="store_true",
                   help="include raw completions in JSON output")
    p.add_argument("--version", action="version", version=f"leakit {__version__}")
    return p


def _read_documents(args) -> list[tuple[str, str]]:
    """Return (id, text) pairs from files, --text, or stdin."""
    docs: list[tuple[str, str]] = []
    for path in args.documents:
        text = Path(path).read_text(encoding="utf-8", errors="replace")
        docs.append((path, text))
    for i, t in enumerate(args.text):
        docs.append((f"--text[{i}]", t))
    if not docs and not sys.stdin.isatty():
        data = sys.stdin.read()
        if data.strip():
            docs.append(("<stdin>", data))
    return docs


def _make_scorer(args) -> LeakIt:
    return LeakIt(
        model=args.model,
        base_url=args.base_url,
        api_key_env=args.api_key_env,
        n_samples=args.samples,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        mode=args.mode,
        concurrency=args.concurrency,
        n_per_request=args.n_per_request,
        statistic=args.statistic,
        k=args.k,
        prefix_chars=args.prefix_chars,
    )


def _print_table(results: list[ScoreResult], percentiles: dict[str, float] | None) -> None:
    name_w = max([len(r.document_id) for r in results] + [8])
    header = f"{'document':<{name_w}}  {'score':>7}  {'samples':>7}"
    if percentiles is not None:
        header += f"  {'pctile':>7}"
    print(header)
    print("-" * len(header))
    for r in results:
        line = f"{r.document_id:<{name_w}}  {r.score:>7.4f}  {r.n_returned:>3}/{r.n_requested:<3}"
        if percentiles is not None:
            pv = percentiles.get(r.document_id, float("nan"))
            line += f"  {pv:>6.1f}%"
        print(line)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    docs = _read_documents(args)
    if not docs:
        print("error: no documents given (pass file paths, --text, or pipe stdin)", file=sys.stderr)
        return 2

    try:
        scorer = _make_scorer(args)
    except (RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    results = [scorer.score(text, document_id=doc_id) for doc_id, text in docs]

    percentiles: dict[str, float] | None = None
    if args.calibrate:
        import glob

        paths: list[str] = []
        for pattern in args.calibrate.split(","):
            paths.extend(sorted(glob.glob(pattern.strip())))
        if not paths:
            print(f"error: --calibrate matched no files: {args.calibrate!r}", file=sys.stderr)
            return 2
        baseline_scores = [
            scorer.score(Path(p).read_text(encoding="utf-8", errors="replace"),
                         document_id=p).score
            for p in paths
        ]
        percentiles = {r.document_id: percentile_of(r.score, baseline_scores) for r in results}

    if args.json:
        out = [r.as_dict(include_completions=args.show_completions) for r in results]
        if percentiles is not None:
            for d in out:
                d["percentile_vs_baseline"] = percentiles.get(d["document"])
        print(json.dumps(out, indent=2))
    else:
        _print_table(results, percentiles)

    if any(r.n_returned == 0 for r in results):
        print("warning: some documents returned no completions (check model/endpoint/key)",
              file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
