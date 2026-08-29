"""The standalone HTML report."""

from __future__ import annotations

from pathlib import Path

import pytest
from ragrank import evaluate
from ragrank.dataset import from_dict
from ragrank.llm import FakeLLM
from ragrank.metric import (
    exact_match,
    response_conciseness,
    response_relevancy,
)


def run(
    responses: list[str] | None = None,
    metrics: list | None = None,
    questions: list[str] | None = None,
) -> object:
    data = from_dict(
        {
            "question": questions or ["who wrote it", "when"],
            "context": [["Ada wrote it."], ["It was 1843."]],
            "response": ["Ada", "1843"],
        },
        return_as_dataset=True,
    )
    return evaluate(
        data,
        metrics=metrics or [response_relevancy],
        llm=FakeLLM(responses=responses or ["0.8", "0.4"]),
    )


def test_the_report_is_a_complete_document() -> None:
    html = run().to_html()

    assert html.startswith("<!DOCTYPE html>")
    assert html.rstrip().endswith("</html>")
    assert "<title>ragrank report</title>" in html


def test_it_carries_no_external_assets() -> None:
    """One file that opens anywhere, with nothing to fetch."""
    html = run().to_html()

    for marker in ("http://", "https://", "<script", "src="):
        assert marker not in html


def test_the_metric_and_its_score_appear() -> None:
    html = run().to_html()

    assert "Response Relevancy" in html
    assert "0.600" in html


def test_the_cost_tier_is_shown() -> None:
    assert "llm" in run().to_html()


def test_questions_and_responses_appear() -> None:
    html = run().to_html()

    assert "who wrote it" in html
    assert "Ada" in html


def test_the_title_can_be_set() -> None:
    html = run().to_html(title="nightly eval")
    assert "<title>nightly eval</title>" in html


def test_it_writes_to_a_path(tmp_path: Path) -> None:
    path = tmp_path / "report.html"
    html = run().to_html(path)

    assert path.read_text(encoding="utf-8") == html


def test_html_in_the_data_is_escaped() -> None:
    """A response is untrusted text, not markup."""
    result = run(questions=["<script>alert(1)</script>", "safe"])
    html = result.to_html()

    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;" in html


def test_a_long_field_is_folded_into_a_disclosure() -> None:
    long_question = "why " * 200
    html = run(questions=[long_question, "short"]).to_html()

    assert "<details><summary>" in html
    assert "..." in html


def test_an_unscored_row_says_so_rather_than_showing_zero() -> None:
    """Keyed on the prompt, so the failing row is the same every run."""
    data = from_dict(
        {
            "question": ["good", "bad"],
            "context": [["c"], ["c"]],
            "response": ["r", "r"],
        },
        return_as_dataset=True,
    )
    llm = FakeLLM(
        response_fn=lambda prompt: "0.9"
        if "good" in prompt
        else "not a number"
    )
    result = evaluate(
        data,
        metrics=[
            response_relevancy.model_copy(update={"max_retries": 0})
        ],
        llm=llm,
    )
    html = result.to_html()

    assert "n/a" in html
    assert "why" in html


def test_a_threshold_produces_a_verdict() -> None:
    gated = response_relevancy.model_copy(update={"threshold": 0.5})
    html = run(metrics=[gated]).to_html()

    assert "PASS" in html or "FAIL" in html
    assert "Pass rate" in html


def test_several_metrics_each_get_a_column() -> None:
    html = run(
        responses=["0.8"],
        metrics=[response_relevancy, response_conciseness],
    ).to_html()

    assert html.count("Response Relevancy") >= 1
    assert html.count("Response Conciseness") >= 1


def test_a_deterministic_metric_reports_the_free_tier() -> None:
    data = from_dict(
        {
            "question": ["q"],
            "context": [["c"]],
            "response": ["Ada"],
            "reference": ["Ada"],
        },
        return_as_dataset=True,
    )
    html = evaluate(
        data, metrics=[exact_match], llm=FakeLLM()
    ).to_html()

    assert "free" in html
    assert "Exact Match" in html


def test_the_header_counts_rows_metrics_and_tokens() -> None:
    html = run().to_html()

    assert "2 rows" in html
    assert "1 metrics" in html
    assert "tokens" in html


def test_a_result_without_per_row_detail_still_renders() -> None:
    result = run()
    trimmed = result.model_copy(update={"results": None})
    html = trimmed.to_html()

    assert "Metrics" in html
    assert "<h2>Rows</h2>" not in html


def test_the_cli_can_write_a_report(tmp_path: Path) -> None:
    """`ragrank eval --html` is the point of having a CLI at all."""
    import json

    from ragrank.cli import main

    dataset = tmp_path / "data.json"
    dataset.write_text(
        json.dumps({
            "question": ["q"],
            "context": [["c"]],
            "response": ["r"],
            "reference": ["r"],
        }),
        encoding="utf-8",
    )
    config = tmp_path / "ragrank.json"
    config.write_text(
        json.dumps({
            "data": {
                "question": ["q"],
                "context": [["c"]],
                "response": ["r"],
                "reference": ["r"],
            },
            "metrics": ["exact_match"],
        }),
        encoding="utf-8",
    )
    report = tmp_path / "report.html"

    code = main(["eval", str(config), "--html", str(report)])

    assert code == 0
    assert report.read_text(encoding="utf-8").startswith(
        "<!DOCTYPE html>"
    )
