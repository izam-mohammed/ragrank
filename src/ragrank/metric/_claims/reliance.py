"""Is retrieval actually doing anything?

Faithfulness asks whether the answer is grounded in the context.
Correctness asks whether it is right. Neither answers the question a
team asks after wiring up a retriever: *is the retriever earning its
keep, or would the model have got this right anyway?*

RAGChecker calls the gap self-knowledge -- correct claims the model
produced that the retrieved context does not support. A system with high
self-knowledge is answering from what the model already knew, which
looks fine on a benchmark and falls over the moment you point it at
private data the model has never seen.

This reports the complement, so that higher stays better like every
other metric here: the share of correct claims retrieval actually
supplied. `metadata["self_knowledge"]` carries RAGChecker's number for
anyone who wants it in the original direction.
"""

from __future__ import annotations

from time import perf_counter
from typing import Any

from ragrank.dataset import DataNode
from ragrank.metric._claims.base import ClaimMetric
from ragrank.metric.base import MetricResult


class ContextReliance(ClaimMetric):
    """What share of the correct answer came from retrieval.

    Each claim in the response is checked twice: against the reference,
    to see whether it is correct, and against the retrieved context, to
    see whether retrieval supplied it. The score is the share of the
    *correct* claims that the context supports.

    Claims that are wrong are ignored rather than counted against the
    retriever -- a hallucination is faithfulness's problem, and mixing
    the two produces a number that moves for two unrelated reasons.

    A low score with good correctness is the interesting case: the
    pipeline works, but the model is carrying it, and it will stop
    working on documents the model has not memorised.

    Costs one extraction call plus two verifications per claim, so it is
    a diagnostic to run on a sample rather than a gate to run on
    everything.
    """

    @property
    def name(self) -> str:
        """Get the name for the metric.

        Returns:
            str: The name of the metric.
        """
        return "Context Reliance"

    @property
    def required_columns(self) -> set[str]:
        """Fields this metric needs.

        Returns:
            set[str]: `context`, `response` and `reference` -- without
                ground truth there is no way to tell a correct claim
                from an invented one.
        """
        return {"context", "response", "reference"}

    def claim_text(self, data: DataNode) -> str:
        """Decompose the response.

        Args:
            data (DataNode): The row being scored.

        Returns:
            str: The response text.
        """
        return data.response

    def source_text(self, data: DataNode) -> str:
        """Check support against the retrieved context.

        Args:
            data (DataNode): The row being scored.

        Returns:
            str: The joined context chunks.
        """
        return "\n\n".join(data.context)

    def score(self, data: DataNode) -> MetricResult:
        """Attribute the correct part of the answer to retrieval.

        Args:
            data (DataNode): The input data to be used for scoring.

        Returns:
            MetricResult: The share of correct claims the context
                supports, with per claim verdicts in
                `metadata["claims"]`.
        """
        started = perf_counter()

        def abstain(reason: str, **extra: Any) -> MetricResult:  # noqa: ANN401
            return MetricResult(
                datanode=data,
                metric=self,
                score=None,
                error=reason,
                metadata=dict(extra),
                process_time=perf_counter() - started,
            )

        reference = (data.reference or "").strip()
        if not reference:
            return abstain("no reference answer to check against")

        context = self.source_text(data)
        if not context.strip():
            return abstain("no context to attribute the answer to")

        claims = self.extract_claims(self.claim_text(data))
        if not claims:
            return abstain(
                "no verifiable claims were found", claims=[]
            )

        verdicts: list[dict[str, Any]] = []
        for claim in claims:
            correct = self.verify_claim(claim, reference)
            supported = (
                self.verify_claim(claim, context)
                if correct
                else None
            )
            verdicts.append({
                "claim": claim,
                "correct": correct,
                "supported": supported,
            })

        correct_claims = [
            item for item in verdicts if item["correct"]
        ]
        attributable = [
            item
            for item in correct_claims
            if item["supported"] is not None
        ]
        metadata: dict[str, Any] = {
            "claims": verdicts,
            "claim_count": len(claims),
            "correct_count": len(correct_claims),
        }

        if not attributable:
            return abstain(
                "no correct claim could be attributed", **metadata
            )

        from_context = sum(
            item["supported"] for item in attributable
        )
        reliance = from_context / len(attributable)
        metadata["self_knowledge"] = 1.0 - reliance

        return MetricResult(
            datanode=data,
            metric=self,
            score=reliance,
            metadata=metadata,
            process_time=perf_counter() - started,
        )


context_reliance = ContextReliance()
