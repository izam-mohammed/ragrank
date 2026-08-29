"""Regression tests for defects listed in PLAN.md section 1.

Each test is named for the defect number it pins down. Every one of these
fails on the commit before the Phase 0 fixes.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
from ragrank.dataset import DataNode, Dataset
from ragrank.llm import FakeLLM
from ragrank.metric import CustomInstruct, InstructConfig
from ragrank.metric._response_related.relevancy import ResponseRelevancy
from ragrank.prompt import Prompt
from ragrank.utils.common import eval_cell


def test_defect_2_dataset_validates_context_length() -> None:
    """Mismatched context length must fail at construction, not on access."""
    with pytest.raises(ValueError, match="context"):
        Dataset(
            question=["a", "b"],
            context=[["c"]],
            response=["r1", "r2"],
        )


def test_defect_2_balanced_dataset_still_constructs() -> None:
    """The tightened validator must not reject valid datasets."""
    dataset = Dataset(
        question=["a", "b"],
        context=[["c1"], ["c2"]],
        response=["r1", "r2"],
    )
    assert len(dataset) == 2


def test_defect_3_custom_instruct_keeps_instructions() -> None:
    """NON_BINARY metrics must not discard the user's instructions."""
    config = InstructConfig(
        metric_type=__import__(
            "ragrank.metric.base", fromlist=["MetricType"]
        ).MetricType.NON_BINARY,
        name="Politeness",
        instructions="SENTINEL_INSTRUCTION_TEXT",
        input_fields=["question"],
    )
    metric = CustomInstruct(config=config, llm=FakeLLM())
    assert "SENTINEL_INSTRUCTION_TEXT" in metric.prompt.instructions


def test_defect_4_process_time_is_positive() -> None:
    """process_time must be a duration, not a negative number."""
    metric = ResponseRelevancy(llm=FakeLLM(responses=["0.5"]))
    result = metric.score(
        DataNode(question="q", context=["c"], response="r")
    )
    assert result.process_time is not None
    assert result.process_time >= 0


def test_defect_9_get_examples_returns_a_list() -> None:
    """get_examples is annotated -> list[Example] and must return one."""
    prompt = Prompt(
        name="P",
        instructions="i",
        examples=[{"question": "a", "out": "b"}],
        input_keys=["question"],
        output_key="out",
    )
    assert isinstance(prompt.get_examples(), list)
    assert isinstance(prompt.get_examples(1), list)


def test_defect_14_makefile_integration_dir_exists() -> None:
    """The Makefile must point at a directory that exists."""
    makefile = Path(__file__).parents[2] / "Makefile"
    line = next(
        entry
        for entry in makefile.read_text().splitlines()
        if entry.startswith("INTEGRATION_TEST_DIR")
    )
    target = line.split("?=")[1].strip()
    assert (makefile.parent / target).is_dir(), target


def test_defect_15_core_requires_only_pydantic() -> None:
    """Unused and harmful packages must not be required dependencies."""
    pyproject = (
        Path(__file__).parents[2] / "pyproject.toml"
    ).read_text()
    required = pyproject.split("dependencies = [")[1].split("]")[0]
    for banned in (
        "pathlib",
        "urllib3",
        "requests-toolbelt",
        "requests>",
    ):
        assert banned not in required, banned


def test_defect_16_py_typed_marker_is_shipped() -> None:
    """Type hints are useless to consumers without the marker."""
    import ragrank

    marker = Path(ragrank.__file__).parent / "py.typed"
    assert marker.is_file()


@pytest.mark.parametrize(
    ("cell", "expected"),
    [
        ("['a', 'b']", ["a", "b"]),
        ('["a", "b"]', ["a", "b"]),
        ("[]", []),
        ("['it\\'s, complicated', 'b']", ["it's, complicated", "b"]),
        ("plain string", "plain string"),
        ("[not a literal", "[not a literal"),
        (["already", "a", "list"], ["already", "a", "list"]),
    ],
)
def test_defect_17_eval_cell_parses_safely(
    cell: str | list[str], expected: str | list[str]
) -> None:
    """eval_cell must use a real parser, not string slicing."""
    assert eval_cell(cell) == expected


def test_defect_18_no_raise_from_exception_class() -> None:
    """`raise X from ValueError` chains from a class, which is a misuse."""
    src = Path(__file__).parents[2] / "src"
    offenders = [
        f"{path.name}:{number}"
        for path in src.rglob("*.py")
        for number, line in enumerate(
            path.read_text().splitlines(), start=1
        )
        if line.strip().endswith(
            ("from ValueError", "from TypeError")
        )
        or line.strip().endswith("from ModuleNotFoundError")
    ]
    assert not offenders, offenders


def test_core_imports_without_optional_dependencies() -> None:
    """The core must import with pandas, tqdm and datasets blocked."""
    script = (
        "import sys\n"
        "for name in ('pandas', 'tqdm', 'datasets'):\n"
        "    sys.modules[name] = None\n"
        "import ragrank\n"
        "from ragrank.dataset import Dataset\n"
        "from ragrank.llm import FakeLLM\n"
        "from ragrank.metric import response_relevancy\n"
        "print('ok')\n"
    )
    completed = subprocess.run(  # noqa: S603
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "ok" in completed.stdout


def test_fake_llm_needs_no_credentials() -> None:
    """The library must be exercisable end to end with no API key."""
    llm = FakeLLM(responses=["0.8", "0.2"])
    assert [llm.generate_text("x").response for _ in range(3)] == [
        "0.8",
        "0.2",
        "0.8",
    ]
    assert llm.prompts == ["x", "x", "x"]
