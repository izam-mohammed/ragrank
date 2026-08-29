"""The library must log, never print.

A library that writes to stdout corrupts the output of any program that
pipes it, and gives the caller no way to turn it off. Everything
diagnostic goes through `logging`, which the caller controls.

Ruff's T20 rule already enforces this on `src` in CI. These tests are
the belt to that braces: T20 works on source text, and these walk the
AST and the actual runtime.
"""

from __future__ import annotations

import ast
import io
from contextlib import redirect_stdout
from pathlib import Path

import pytest
from ragrank import evaluate
from ragrank.dataset import Dataset
from ragrank.evaluation import RunConfig
from ragrank.llm import FakeLLM
from ragrank.metric import exact_match, faithfulness

SRC = Path(__file__).parents[2] / "src" / "ragrank"


def source_files() -> list[Path]:
    """Every shipped Python module."""
    return [
        path
        for path in sorted(SRC.rglob("*.py"))
        if path.name != "_version.py"
    ]


def test_there_are_source_files_to_check() -> None:
    """Guard against the glob silently matching nothing."""
    assert len(source_files()) > 20


@pytest.mark.parametrize(
    "path", source_files(), ids=lambda p: p.name
)
def test_no_executable_print_calls(path: Path) -> None:
    """No module may call print().

    Walks the AST, so a `print(...)` inside a docstring example -- which
    is guidance for the caller, not the library writing to stdout --
    does not count, and a real call cannot hide inside one.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"), str(path))
    offenders = [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "print"
    ]
    assert not offenders, f"print() at {path.name}:{offenders}"


@pytest.mark.parametrize(
    "path", source_files(), ids=lambda p: p.name
)
def test_no_direct_stdout_writes(path: Path) -> None:
    """No module may reach for sys.stdout or sys.stderr directly."""
    text = path.read_text(encoding="utf-8")
    for banned in ("sys.stdout", "sys.stderr"):
        assert banned not in text, f"{banned} in {path.name}"


def test_a_full_run_prints_nothing() -> None:
    """The runtime proof: evaluate() writes nothing to stdout.

    tqdm deliberately writes its bar to stderr, so a progress bar does
    not interfere with piping results.
    """
    dataset = Dataset(
        question=["q"],
        context=[["c"]],
        response=["r"],
        reference=["r"],
    )
    llm = FakeLLM(
        response_fn=lambda p: '["a claim"]'
        if p.startswith("Claim Extraction")
        else "A"
    )

    captured = io.StringIO()
    with redirect_stdout(captured):
        result = evaluate(
            dataset,
            llm=llm,
            metrics=[faithfulness, exact_match],
            run_config=RunConfig(show_progress=True, max_workers=1),
        )

    assert captured.getvalue() == ""
    assert result.scores[0][0] is not None


def test_diagnostics_go_to_the_logger(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """An unusable judge answer is logged, not printed."""
    dataset = Dataset(
        question=["q"], context=[["c"]], response=["r"]
    )
    from ragrank.metric import response_relevancy

    captured = io.StringIO()
    with caplog.at_level("WARNING"), redirect_stdout(captured):
        evaluate(
            dataset,
            llm=FakeLLM(responses=["banana"]),
            metrics=[response_relevancy],
            run_config=RunConfig(show_progress=False, max_workers=1),
        )

    assert captured.getvalue() == ""
    assert any(
        "unusable answer" in record.message
        for record in caplog.records
    )
