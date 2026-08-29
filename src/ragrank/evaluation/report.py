"""A self-contained HTML report for a finished run.

A DataFrame is the right shape for someone holding a notebook. It is
the wrong shape for the person who asked whether the change was safe to
ship, who wants to see which rows failed and what the judge said about
them, and who is not going to install anything to find out.

One file, no assets, no server, no network. Open it or attach it to a
pull request.
"""

from __future__ import annotations

from html import escape
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover
    from ragrank.evaluation.outputs import EvalResult

#: Deliberately plain. A report that needs a design system is a report
#: that stops rendering the moment the CDN blinks.
STYLE = """
:root { color-scheme: light dark; }
body { font-family: ui-sans-serif, system-ui, -apple-system, sans-serif;
       margin: 2rem auto; max-width: 70rem; padding: 0 1rem;
       line-height: 1.5; }
h1 { font-size: 1.4rem; margin-bottom: 0.25rem; }
.meta { opacity: 0.7; font-size: 0.85rem; margin-bottom: 1.5rem; }
table { border-collapse: collapse; width: 100%; font-size: 0.9rem;
        margin-bottom: 2rem; }
th, td { text-align: left; padding: 0.45rem 0.6rem;
         border-bottom: 1px solid rgba(128,128,128,0.3);
         vertical-align: top; }
th { font-weight: 600; white-space: nowrap; }
td.num { text-align: right; font-variant-numeric: tabular-nums;
         white-space: nowrap; }
.pass { color: #157f3d; font-weight: 600; }
.fail { color: #b3261e; font-weight: 600; }
.none { opacity: 0.5; font-style: italic; }
.tier { font-size: 0.75rem; opacity: 0.65; }
details { margin: 0; }
summary { cursor: pointer; }
.wrap { max-width: 26rem; overflow-wrap: anywhere; }
.scroll { overflow-x: auto; }
"""


def _number(value: float | None, digits: int = 3) -> str:
    """Format a number, or say plainly that there isn't one."""
    if value is None:
        return '<span class="none">n/a</span>'
    return f"{value:.{digits}f}"


def _verdict(passed: bool | None) -> str:
    """Format a pass/fail verdict."""
    if passed is None:
        return '<span class="none">-</span>'
    return (
        '<span class="pass">PASS</span>'
        if passed
        else '<span class="fail">FAIL</span>'
    )


def _clip(text: str, limit: int = 300) -> str:
    """Escape text, and fold anything long into a disclosure."""
    if len(text) <= limit:
        return escape(text)
    return (
        "<details><summary>"
        + escape(text[:limit].rstrip())
        + " ...</summary>"
        + escape(text)
        + "</details>"
    )


def _summary_table(result: EvalResult) -> str:
    """Build the per-metric summary table."""
    tiers = {
        metric.name: metric.cost_tier.value
        for metric in result.metrics
    }

    rows = []
    for item in result.summary():
        stderr = (
            f" &plusmn; {item.stderr:.3f}"
            if item.stderr is not None
            else ""
        )
        rows.append(
            "<tr>"
            f"<td>{escape(item.name)}<br>"
            f'<span class="tier">{tiers.get(item.name, "")}</span>'
            "</td>"
            f'<td class="num">{_number(item.value)}{stderr}</td>'
            f'<td class="num">{item.scored}/{item.count}</td>'
            f'<td class="num">{_number(item.pass_rate, 2)}</td>'
            f'<td class="num">{_verdict(item.passed)}</td>'
            "</tr>"
        )

    return (
        "<h2>Metrics</h2><div class='scroll'><table><thead><tr>"
        "<th>Metric</th><th>Score</th><th>Scored</th>"
        "<th>Pass rate</th><th>Verdict</th>"
        "</tr></thead><tbody>" + "".join(rows) + "</tbody></table></div>"
    )


def _cell(entry: Any) -> str:  # noqa: ANN401
    """Render one score cell, with the judge's reasoning folded in."""
    if entry is None:
        return '<td class="num"><span class="none">-</span></td>'

    score = _number(entry.score)
    detail = entry.error or entry.reason
    if not detail:
        return f'<td class="num">{score}</td>'

    return (
        f'<td class="num">{score}'
        f"<details><summary>why</summary>{escape(str(detail))}"
        "</details></td>"
    )


def _rows_table(result: EvalResult) -> str:
    """Build the per-row detail table."""
    if result.results is None:
        return ""

    names = [metric.name for metric in result.metrics]
    header = "".join(f"<th>{escape(name)}</th>" for name in names)

    rows = []
    for index, node in enumerate(result.dataset):
        cells = "".join(
            _cell(
                result.results[metric][index]
                if index < len(result.results[metric])
                else None
            )
            for metric in range(len(names))
        )
        rows.append(
            "<tr>"
            f'<td class="num">{index + 1}</td>'
            f'<td class="wrap">{_clip(node.question)}</td>'
            f'<td class="wrap">{_clip(node.response)}</td>'
            f"{cells}</tr>"
        )

    return (
        "<h2>Rows</h2><div class='scroll'><table><thead><tr>"
        "<th>#</th><th>Question</th><th>Response</th>"
        f"{header}</tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table></div>"
    )


def _document(title: str, body: str) -> str:
    """Wrap rendered sections in a complete HTML document."""
    return (
        "<!DOCTYPE html><html lang='en'><head>"
        "<meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,"
        "initial-scale=1'>"
        f"<title>{escape(title)}</title>"
        f"<style>{STYLE}</style></head><body>"
        f"<h1>{escape(title)}</h1>"
        f"{body}</body></html>"
    )


def _meta(result: EvalResult) -> str:
    """One line describing how the run went."""
    verdict = (
        ""
        if result.passed is None
        else f" &middot; {_verdict(result.passed)}"
    )
    return (
        f"{len(result.dataset)} rows &middot; "
        f"{len(result.metrics)} metrics &middot; "
        f"judged by {escape(result.llm.name)} &middot; "
        f"{result.response_time:.2f}s &middot; "
        f"{result.usage.total_tokens:,} tokens"
        f"{verdict}"
    )


def combined_html(
    entries: list[tuple[str, EvalResult]],
    title: str = "ragrank report",
) -> str:
    """Render several runs into one document, one section each.

    Used by the pytest plugin, where each eval test contributes a run
    and the useful artefact is all of them together.

    Args:
        entries (list[tuple[str, EvalResult]]): Named runs, in order.
        title (str): Title for the document.

    Returns:
        str: A complete HTML document.
    """
    if not entries:
        return _document(title, "<p class='meta'>No evaluations ran.</p>")

    sections = [
        f"<h2>{escape(name)}</h2>"
        f"<p class='meta'>{_meta(result)}</p>"
        + _summary_table(result)
        + _rows_table(result)
        for name, result in entries
    ]
    return _document(title, "".join(sections))


def to_html(result: EvalResult, title: str = "ragrank report") -> str:
    """Render a finished run as a standalone HTML document.

    Args:
        result (EvalResult): The run to render.
        title (str): Title for the document.

    Returns:
        str: A complete HTML document, with no external assets.
    """
    return _document(
        title,
        f"<p class='meta'>{_meta(result)}</p>"
        + _summary_table(result)
        + _rows_table(result),
    )
