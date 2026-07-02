"""End-to-end CLI tests with an injected fake client (no network)."""

import json

import pytest

from leakit import cli
from leakit.core import LeakIt


@pytest.fixture
def patch_scorer(monkeypatch, fake_client):
    """Replace cli._make_scorer with one that injects a scripted fake client."""
    def _factory(args):
        fc = fake_client(
            script={
                "MEMBER": ["born 1809 hardin county"],  # identical -> full concentration
                "NOVEL": ["sky blue today", "i wonder if", "banana telephone qux"],
            },
            default=["a b c", "a b d"],
        )
        return LeakIt(
            model=args.model, n_samples=args.samples, statistic=args.statistic,
            prefix_chars=args.prefix_chars, mode=args.mode, client=fc, api_key="k",
        )
    monkeypatch.setattr(cli, "_make_scorer", _factory)


def test_cli_table_output(patch_scorer, tmp_path, capsys):
    doc = tmp_path / "suspect.txt"
    doc.write_text("MEMBER text here")
    rc = cli.main(["--model", "m", str(doc)])
    out = capsys.readouterr().out
    assert rc == 0
    assert "suspect.txt" in out
    assert "score" in out


def test_cli_json_output(patch_scorer, tmp_path, capsys):
    doc = tmp_path / "d.txt"
    doc.write_text("MEMBER text")
    rc = cli.main(["--model", "m", "--json", str(doc)])
    out = capsys.readouterr().out
    data = json.loads(out)
    assert rc == 0
    assert data[0]["document"].endswith("d.txt")
    assert data[0]["n_returned"] == 16
    assert 0.0 <= data[0]["score"] <= 1.0


def test_cli_member_scores_above_novel(patch_scorer, tmp_path, capsys):
    m = tmp_path / "m.txt"; m.write_text("MEMBER passage")
    nvl = tmp_path / "n.txt"; nvl.write_text("NOVEL passage")
    cli.main(["--model", "m", "--json", str(m), str(nvl)])
    data = json.loads(capsys.readouterr().out)
    by = {d["document"].split("/")[-1]: d["score"] for d in data}
    assert by["m.txt"] > by["n.txt"]


def test_cli_calibrate_percentile(patch_scorer, tmp_path, capsys):
    suspect = tmp_path / "suspect.txt"; suspect.write_text("MEMBER passage")
    clean1 = tmp_path / "c1.txt"; clean1.write_text("NOVEL one")
    clean2 = tmp_path / "c2.txt"; clean2.write_text("NOVEL two")
    rc = cli.main(["--model", "m", "--json",
                   "--calibrate", f"{clean1},{clean2}", str(suspect)])
    data = json.loads(capsys.readouterr().out)
    assert rc == 0
    # MEMBER concentrates more than the NOVEL baseline -> high percentile
    assert data[0]["percentile_vs_baseline"] == 100.0


def test_cli_no_documents_errors(monkeypatch, capsys):
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    rc = cli.main(["--model", "m"])
    assert rc == 2
    assert "no documents" in capsys.readouterr().err


def test_cli_stdin(patch_scorer, monkeypatch, capsys):
    import io
    monkeypatch.setattr("sys.stdin", io.StringIO("MEMBER from stdin"))
    monkeypatch.setattr("sys.stdin.isatty", lambda: False, raising=False)
    rc = cli.main(["--model", "m", "--json"])
    data = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert data[0]["document"] == "<stdin>"
