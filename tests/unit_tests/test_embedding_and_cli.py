"""Tests for embeddings, semantic similarity, and the CLI."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from ragrank.cli import (
    EXIT_BAD_USAGE,
    EXIT_FAILED_THRESHOLD,
    EXIT_OK,
    load_config,
    main,
    resolve_metrics,
)
from ragrank.dataset import DataNode
from ragrank.embedding import (
    BaseEmbedding,
    FakeEmbedding,
    cosine_similarity,
)
from ragrank.metric import SemanticSimilarity, exact_match

CSV = (
    "question,context,response,reference\n"
    "What is the capital of France?,\"['Paris is the capital.']\","
    "Paris,Paris\n"
    "What is the capital of Japan?,\"['Tokyo is the capital.']\","
    "Kyoto,Tokyo\n"
)


def node(response: str, reference: str | None = None) -> DataNode:
    """Build a one-off data node."""
    return DataNode(
        question="q",
        context=["c"],
        response=response,
        reference=reference,
    )


# --------------------------- cosine ---------------------------


@pytest.mark.parametrize(
    ("left", "right", "expected"),
    [
        ([1.0, 0.0], [1.0, 0.0], 1.0),
        ([1.0, 0.0], [0.0, 1.0], 0.0),
        ([1.0, 0.0], [-1.0, 0.0], 0.0),
        ([0.0, 0.0], [1.0, 1.0], 0.0),
        ([1.0, 1.0], [2.0, 2.0], 1.0),
    ],
)
def test_cosine_similarity(
    left: list[float], right: list[float], expected: float
) -> None:
    """Including the clamp: opposite vectors score 0, not -1."""
    assert cosine_similarity(left, right) == pytest.approx(expected)


def test_cosine_rejects_mismatched_lengths() -> None:
    """Comparing different-sized vectors is a bug, not a zero."""
    with pytest.raises(ValueError, match="length"):
        cosine_similarity([1.0], [1.0, 2.0])


# --------------------------- embeddings ---------------------------


def test_fake_embedding_is_deterministic() -> None:
    """The same text must always embed identically."""
    model = FakeEmbedding()
    assert model.embed_text("hello") == model.embed_text("hello")


def test_fake_embedding_separates_different_text() -> None:
    """Different text must not collide."""
    model = FakeEmbedding()
    assert model.embed_text("hello") != model.embed_text("goodbye")


def test_embedding_dimensions_are_configurable() -> None:
    """Length is a setting."""
    assert len(FakeEmbedding(dimensions=32).embed_text("x")) == 32


def test_batch_embed_matches_single() -> None:
    """The default batching is just a loop, and must agree."""
    model = FakeEmbedding()
    batch = model.embed(["a", "b"])
    assert batch == [model.embed_text("a"), model.embed_text("b")]


def test_similarity_of_identical_text_is_one() -> None:
    """Whitespace and case are normalised away."""
    model = FakeEmbedding()
    assert model.similarity("Paris", " paris ") == pytest.approx(1.0)


def test_embedding_repr_is_its_name() -> None:
    """Readable in logs."""
    assert repr(FakeEmbedding()) == "Fake Embedding"


def test_custom_embedding_subclass() -> None:
    """Anyone can implement the interface."""

    class Constant(BaseEmbedding):
        @property
        def name(self) -> str:
            return "Constant"

        def embed_text(self, text: str) -> list[float]:
            return [1.0, 0.0]

    assert Constant().similarity("a", "b") == pytest.approx(1.0)


# ---------------------- semantic similarity ----------------------


def test_semantic_similarity_scores_identical_text() -> None:
    """Same meaning, same words: 1.0."""
    metric = SemanticSimilarity(embedding=FakeEmbedding())
    assert metric.score(node("Paris", "Paris")).score == 1.0


def test_semantic_similarity_needs_a_reference() -> None:
    """No ground truth, no score."""
    metric = SemanticSimilarity(embedding=FakeEmbedding())
    assert metric.score(node("Paris")).score is None


def test_semantic_similarity_without_a_model_says_so() -> None:
    """An embedding model is a deployment choice, so there is no
    default -- but the failure must be actionable."""
    result = SemanticSimilarity().score(node("a", "b"))
    assert result.score is None
    assert "embedding" in result.error


def test_semantic_similarity_declares_its_columns() -> None:
    """For up-front validation."""
    assert SemanticSimilarity().required_columns == {
        "response",
        "reference",
    }


def test_semantic_similarity_makes_no_llm_calls() -> None:
    """It is in the cheap tier, not the judged one."""
    metric = SemanticSimilarity(embedding=FakeEmbedding())
    assert metric.llm is None
    assert metric.prompt is None


# --------------------------- CLI ---------------------------


@pytest.fixture
def project(tmp_path: Path) -> Path:
    """A directory with a dataset and a config."""
    (tmp_path / "data.csv").write_text(CSV, encoding="utf-8")
    (tmp_path / "ragrank.json").write_text(
        json.dumps({
            "dataset": "data.csv",
            "metrics": [
                "exact_match",
                {"name": "token_f1", "threshold": 0.9},
            ],
            "run": {"show_progress": False, "max_workers": 1},
        }),
        encoding="utf-8",
    )
    return tmp_path


def test_eval_returns_one_when_a_threshold_fails(
    project: Path,
) -> None:
    """The CI contract: a failing eval is a non-zero exit."""
    assert (
        main(["eval", str(project / "ragrank.json")])
        == EXIT_FAILED_THRESHOLD
    )


def test_eval_returns_zero_when_thresholds_pass(
    project: Path,
) -> None:
    """And a passing one exits cleanly."""
    config = json.loads(
        (project / "ragrank.json").read_text(encoding="utf-8")
    )
    config["metrics"][1]["threshold"] = 0.1
    (project / "ragrank.json").write_text(
        json.dumps(config), encoding="utf-8"
    )
    assert main(["eval", str(project / "ragrank.json")]) == EXIT_OK


def test_eval_writes_a_result_file(project: Path) -> None:
    """--output gives you something to diff later."""
    out = project / "result.json"
    main([
        "eval",
        str(project / "ragrank.json"),
        "--output",
        str(out),
    ])
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert {"summary", "data", "usage"} <= set(payload)


def test_inline_data_needs_no_csv(tmp_path: Path) -> None:
    """A config can carry its own data."""
    config = tmp_path / "inline.json"
    config.write_text(
        json.dumps({
            "data": {
                "question": ["q"],
                "context": [["c"]],
                "response": ["r"],
                "reference": ["r"],
            },
            "metrics": ["exact_match"],
            "run": {"show_progress": False},
        }),
        encoding="utf-8",
    )
    assert main(["eval", str(config)]) == EXIT_OK


def test_missing_config_is_a_usage_error() -> None:
    """Exit 2, not a traceback."""
    assert main(["eval", "/does/not/exist.json"]) == EXIT_BAD_USAGE


def test_unknown_metric_lists_the_options(tmp_path: Path) -> None:
    """The error must be actionable."""
    config = tmp_path / "bad.json"
    config.write_text(
        json.dumps({
            "data": {
                "question": ["q"],
                "context": [["c"]],
                "response": ["r"],
            },
            "metrics": ["not_a_metric"],
        }),
        encoding="utf-8",
    )
    assert main(["eval", str(config)]) == EXIT_BAD_USAGE
    with pytest.raises(ValueError, match="Unknown metric"):
        resolve_metrics(["not_a_metric"])


def test_config_without_metrics_is_rejected(
    tmp_path: Path,
) -> None:
    """Evaluating nothing is a mistake."""
    config = tmp_path / "empty.json"
    config.write_text(
        json.dumps({
            "data": {
                "question": ["q"],
                "context": [["c"]],
                "response": ["r"],
            },
            "metrics": [],
        }),
        encoding="utf-8",
    )
    assert main(["eval", str(config)]) == EXIT_BAD_USAGE


def test_config_without_data_is_rejected(tmp_path: Path) -> None:
    """So is evaluating nothing."""
    config = tmp_path / "nodata.json"
    config.write_text(
        json.dumps({"metrics": ["exact_match"]}), encoding="utf-8"
    )
    assert main(["eval", str(config)]) == EXIT_BAD_USAGE


def test_resolve_metrics_applies_a_threshold() -> None:
    """A config can gate without touching Python."""
    resolved = resolve_metrics([
        {"name": "exact_match", "threshold": 0.8}
    ])
    assert resolved[0].threshold == 0.8
    assert (
        exact_match.threshold is None
    ), "must not mutate the shared metric"


def test_load_config_rejects_a_non_mapping(tmp_path: Path) -> None:
    """A list is not a config."""
    path = tmp_path / "list.json"
    path.write_text("[1, 2]", encoding="utf-8")
    with pytest.raises(ValueError, match="mapping"):
        load_config(path)


def test_compare_subcommand_diffs_two_results(
    project: Path,
) -> None:
    """The workflow: save two runs, diff them."""
    first, second = project / "a.json", project / "b.json"
    main(["eval", str(project / "ragrank.json"), "-o", str(first)])
    main(["eval", str(project / "ragrank.json"), "-o", str(second)])
    assert main(["compare", str(first), str(second)]) == EXIT_OK


def test_compare_rejects_a_file_that_is_not_a_result(
    project: Path, tmp_path: Path
) -> None:
    """A clear message beats a KeyError."""
    junk = tmp_path / "junk.json"
    junk.write_text("{}", encoding="utf-8")
    assert main(["compare", str(junk), str(junk)]) == EXIT_BAD_USAGE
