"""A command line front end, so an eval is a file you can review.

    ragrank eval ragrank.yaml

A config in the repository is diffable, reviewable in a pull request,
and editable by someone who does not write Python. That is a different
audience from the SDK, and a cheap one to serve given the core already
exists.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

from ragrank.dataset import ColumnMap, Dataset, from_csv, from_dict
from ragrank.evaluation import RunConfig, evaluate
from ragrank.evaluation.outputs import EvalResult
from ragrank.metric import BaseMetric

logger = logging.getLogger(__name__)

EXIT_OK = 0
EXIT_FAILED_THRESHOLD = 1
EXIT_BAD_USAGE = 2


def load_config(path: Path) -> dict[str, Any]:
    """Read a YAML or JSON config file.

    YAML needs PyYAML, which is not a dependency -- a JSON config works
    with no extra install.

    Args:
        path (Path): Path to the config file.

    Returns:
        dict[str, Any]: The parsed config.

    Raises:
        ValueError: If the file cannot be parsed.
    """
    text = path.read_text(encoding="utf-8")

    if path.suffix in {".yaml", ".yml"}:
        try:
            import yaml
        except ModuleNotFoundError:
            raise ValueError(
                "Reading YAML needs PyYAML (`pip install pyyaml`). "
                "A .json config works without it."
            ) from None
        parsed = yaml.safe_load(text)
    else:
        parsed = json.loads(text)

    if not isinstance(parsed, dict):
        raise ValueError(
            f"{path} should contain a mapping, not "
            f"{type(parsed).__name__}."
        )
    return parsed


def resolve_metrics(names: list[str]) -> list[BaseMetric]:
    """Turn metric names from a config into metric objects.

    Args:
        names (list[str]): Names as written in the config.

    Returns:
        list[BaseMetric]: The resolved metrics.

    Raises:
        ValueError: If a name is not a known metric.
    """
    import ragrank.metric as registry

    resolved = []
    for entry in names:
        name = entry
        threshold = None
        if isinstance(entry, dict):
            name = entry.get("name", "")
            threshold = entry.get("threshold")

        candidate = getattr(registry, name, None)
        if not isinstance(candidate, BaseMetric):
            available = sorted(
                item
                for item in registry.__all__
                if isinstance(
                    getattr(registry, item, None), BaseMetric
                )
            )
            raise ValueError(
                f"Unknown metric {name!r}. Available: "
                + ", ".join(available)
            )
        if threshold is not None:
            candidate = candidate.model_copy(
                update={"threshold": threshold}
            )
        resolved.append(candidate)
    return resolved


def load_dataset(config: dict[str, Any], root: Path) -> Dataset:
    """Build the dataset described by a config.

    Args:
        config (dict[str, Any]): The parsed config.
        root (Path): Directory the config lives in, for relative paths.

    Returns:
        Dataset: The loaded dataset.

    Raises:
        ValueError: If no usable dataset is described.
    """
    column_map = ColumnMap(**config.get("column_map", {}))

    if "dataset" in config:
        path = root / config["dataset"]
        if not path.exists():
            raise ValueError(f"Dataset file not found: {path}")
        data = from_csv(path, column_map=column_map)
    elif "data" in config:
        data = from_dict(
            config["data"],
            column_map=column_map,
            return_as_dataset=True,
        )
    else:
        raise ValueError(
            "The config needs either `dataset:` (a CSV path) or "
            "`data:` (inline columns)."
        )

    return data if isinstance(data, Dataset) else data.to_dataset()


def run_eval(args: argparse.Namespace) -> int:
    """Run the `eval` subcommand.

    Args:
        args (argparse.Namespace): Parsed arguments.

    Returns:
        int: The process exit code.
    """
    path = Path(args.config)
    config = load_config(path)

    dataset = load_dataset(config, path.parent)
    metrics = resolve_metrics(config.get("metrics", []))
    if not metrics:
        raise ValueError("The config lists no metrics.")

    run_config = RunConfig(**config.get("run", {}))
    result = evaluate(
        dataset, metrics=metrics, run_config=run_config
    )

    _emit(result, args.output, getattr(args, "html", None))

    if result.passed is False:
        logger.error("Evaluation failed its thresholds")
        return EXIT_FAILED_THRESHOLD
    return EXIT_OK


def run_compare(args: argparse.Namespace) -> int:
    """Run the `compare` subcommand.

    Args:
        args (argparse.Namespace): Parsed arguments.

    Returns:
        int: The process exit code.
    """
    sys.stdout.write(
        "compare reads two JSON files written by `ragrank eval "
        "--output`.\n"
    )
    before = json.loads(
        Path(args.baseline).read_text(encoding="utf-8")
    )
    after = json.loads(
        Path(args.candidate).read_text(encoding="utf-8")
    )
    for name in ("baseline", "candidate"):
        payload = before if name == "baseline" else after
        if "summary" not in payload:
            raise ValueError(
                f"The {name} file has no `summary`; was it written "
                "by `ragrank eval --output`?"
            )

    lines = _summary_diff(before["summary"], after["summary"])
    sys.stdout.write("\n".join(lines) + "\n")
    return EXIT_OK


def _summary_diff(
    before: list[dict[str, Any]], after: list[dict[str, Any]]
) -> list[str]:
    """Format a plain diff of two saved summaries."""
    old = {item["name"]: item for item in before}
    lines = []
    for item in after:
        name = item["name"]
        if name not in old:
            continue
        was, now = old[name]["value"], item["value"]
        if was is None or now is None:
            lines.append(f"{name}: n/a")
            continue
        lines.append(
            f"{name}: {was:.3f} -> {now:.3f} ({now - was:+.3f})"
        )
    return lines or ["no shared metrics"]


def _emit(
    result: EvalResult,
    output: str | None,
    html: str | None = None,
) -> None:
    """Write the result to stdout, and to files if asked.

    stdout here is the command's *output*, not logging -- a CLI is
    expected to print its results, and this is the one place in the
    package that writes to it.

    Args:
        result (EvalResult): The result to write.
        output (str | None): Optional path for a JSON copy.
        html (str | None): Optional path for an HTML report.
    """
    for item in result.summary():
        value = "n/a" if item.value is None else f"{item.value:.3f}"
        line = f"{item.name}: {value}"
        if item.stderr is not None:
            line += f" +/- {item.stderr:.3f}"
        if item.passed is not None:
            line += "  [PASS]" if item.passed else "  [FAIL]"
        sys.stdout.write(line + "\n")

    sys.stdout.write(f"\n{result.usage}\n")

    if output:
        Path(output).write_text(
            result.to_json(indent=2), encoding="utf-8"
        )
        sys.stdout.write(f"written to {output}\n")

    if html:
        result.to_html(html)
        sys.stdout.write(f"report written to {html}\n")


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser.

    Returns:
        argparse.ArgumentParser: The parser.
    """
    parser = argparse.ArgumentParser(
        prog="ragrank",
        description="Evaluate a RAG pipeline from a config file.",
    )
    subcommands = parser.add_subparsers(
        dest="command", required=True
    )

    evaluate_cmd = subcommands.add_parser(
        "eval", help="run an evaluation described by a config file"
    )
    evaluate_cmd.add_argument(
        "config", help="path to a .yaml or .json config"
    )
    evaluate_cmd.add_argument(
        "-o", "--output", help="write the full result to this file"
    )
    evaluate_cmd.add_argument(
        "--html", help="write a standalone HTML report to this file"
    )
    evaluate_cmd.set_defaults(handler=run_eval)

    compare_cmd = subcommands.add_parser(
        "compare", help="diff two saved evaluation results"
    )
    compare_cmd.add_argument("baseline")
    compare_cmd.add_argument("candidate")
    compare_cmd.set_defaults(handler=run_compare)

    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point for the `ragrank` command.

    Args:
        argv (list[str] | None): Arguments, defaulting to sys.argv.

    Returns:
        int: The process exit code.
    """
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s: %(message)s"
    )
    args = build_parser().parse_args(argv)
    try:
        return int(args.handler(args))
    except (ValueError, OSError) as error:
        logger.error("%s", error)
        return EXIT_BAD_USAGE


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
