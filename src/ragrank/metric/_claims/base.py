"""Claim decomposition, and metrics built on it.

Judging a whole response in one call gives you a number with no
explanation and no way to see which part went wrong. Decomposing it into
atomic claims and checking each against a source costs more calls but
produces a score that is a *ratio of things you can point at*.

`ClaimMetric` is deliberately a reusable primitive rather than one
metric: extract once, verify once, and several metrics fall out of the
same pass.
"""

from __future__ import annotations

from time import perf_counter
from typing import Any

from ragrank.bridge.pydantic import Field
from ragrank.dataset import DataNode
from ragrank.metric.base import (
    CostTier,
    LLMMetric,
    MetricResult,
    MetricType,
)
from ragrank.metric.parse import parse_list, parse_score
from ragrank.prompt import Prompt
from ragrank.prompt._prompts import (
    CLAIM_EXTRACTION_PROMPT,
    CLAIM_VERIFICATION_PROMPT,
    CLAIM_VERIFICATION_RUBRIC,
)


class ClaimMetric(LLMMetric):
    """Base for metrics that score a ratio of verified claims.

    Subclasses say which text is decomposed and which text it is checked
    against; everything else is shared.

    Attributes:
        extraction_prompt (Prompt): Prompt that splits text into claims.
        verification_prompt (Prompt): Prompt that checks one claim.
        max_claims (int | None): Cap on claims verified per row, so a
            very long response cannot run away with your budget.
    """

    metric_type: MetricType = Field(
        default=MetricType.NON_BINARY,
        description="The type of the metric.",
    )
    prompt: Prompt | None = Field(
        default=CLAIM_VERIFICATION_PROMPT,
        description="The prompt used to verify a single claim.",
    )
    extraction_prompt: Prompt = Field(
        default=CLAIM_EXTRACTION_PROMPT,
        repr=False,
        description="The prompt used to split text into claims.",
    )
    rubric: dict[str, float] | None = Field(
        default_factory=lambda: dict(CLAIM_VERIFICATION_RUBRIC),
        description="Verdict labels for a single claim.",
    )
    max_claims: int | None = Field(
        default=50,
        ge=1,
        repr=False,
        description="Cap on claims verified per row.",
    )

    @property
    def cost_tier(self) -> CostTier:
        """One extraction call plus one call per claim.

        Returns:
            CostTier: Always `CostTier.LLM_HEAVY`.
        """
        return CostTier.LLM_HEAVY

    def claim_text(self, data: DataNode) -> str:
        """The text to decompose into claims.

        Args:
            data (DataNode): The row being scored.

        Returns:
            str: The text to decompose.
        """
        raise NotImplementedError

    def source_text(self, data: DataNode) -> str:
        """The text each claim is checked against.

        Args:
            data (DataNode): The row being scored.

        Returns:
            str: The source text.
        """
        raise NotImplementedError

    def extract_claims(self, text: str) -> list[str]:
        """Split text into atomic, independently checkable claims.

        Args:
            text (str): The text to decompose.

        Returns:
            list[str]: The claims, capped at `max_claims`.
        """
        if not text.strip():
            return []

        llm = self.resolve_llm()
        rendered = self.extraction_prompt.render({"text": text})
        claims = parse_list(llm.generate_text(rendered).response)
        return (
            claims[: self.max_claims]
            if self.max_claims is not None
            else claims
        )

    def verify_claim(self, claim: str, source: str) -> float | None:
        """Check one claim against the source text.

        Args:
            claim (str): The claim to check.
            source (str): The text to check it against.

        Returns:
            float | None: 1.0 if supported, 0.0 if not, None if the
                judge gave no usable answer.
        """
        llm = self.resolve_llm()
        rendered = self.prompt.render({
            "source": source,
            "claim": claim,
        })
        parsed = parse_score(
            llm.generate_text(rendered).response, rubric=self.rubric
        )
        return parsed.score

    def score(self, data: DataNode) -> MetricResult:
        """Decompose, verify, and report the supported ratio.

        A row whose text contains no factual claims is not a failure and
        not a zero -- there was simply nothing to check, so it abstains.

        Args:
            data (DataNode): The input data to be used for scoring.

        Returns:
            MetricResult: The ratio, with per claim verdicts in
                `metadata["claims"]`.
        """
        started = perf_counter()

        source = self.source_text(data)
        if not source.strip():
            return MetricResult(
                datanode=data,
                metric=self,
                score=None,
                error="no source text to verify against",
                process_time=perf_counter() - started,
            )

        claims = self.extract_claims(self.claim_text(data))
        if not claims:
            return MetricResult(
                datanode=data,
                metric=self,
                score=None,
                error="no verifiable claims were found",
                metadata={"claims": []},
                process_time=perf_counter() - started,
            )

        verdicts: list[dict[str, Any]] = []
        for claim in claims:
            verdict = self.verify_claim(claim, source)
            verdicts.append({"claim": claim, "supported": verdict})

        scored = [
            item["supported"]
            for item in verdicts
            if item["supported"] is not None
        ]
        metadata = {
            "claims": verdicts,
            "claim_count": len(claims),
            "verified_count": len(scored),
        }

        if not scored:
            return MetricResult(
                datanode=data,
                metric=self,
                score=None,
                error="no claim could be verified",
                metadata=metadata,
                process_time=perf_counter() - started,
            )

        return MetricResult(
            datanode=data,
            metric=self,
            score=sum(scored) / len(scored),
            metadata=metadata,
            process_time=perf_counter() - started,
        )


class Faithfulness(ClaimMetric):
    """What fraction of the response is supported by the context.

    The core hallucination check, and the most used metric in RAG
    evaluation: it answers "is the model making things up?". A low score
    means the generator invented content the retrieved context does not
    support, regardless of whether that content happens to be true.

    `metadata["claims"]` lists each claim and its verdict, so a bad score
    points at the sentence that caused it.
    """

    @property
    def name(self) -> str:
        """Get the name for the metric.

        Returns:
            str: The name of the metric.
        """
        return "Faithfulness"

    @property
    def required_columns(self) -> set[str]:
        """Fields this metric needs.

        Returns:
            set[str]: `context` and `response`.
        """
        return {"context", "response"}

    def claim_text(self, data: DataNode) -> str:
        """Decompose the response.

        Args:
            data (DataNode): The row being scored.

        Returns:
            str: The response text.
        """
        return data.response

    def source_text(self, data: DataNode) -> str:
        """Check claims against the retrieved context.

        Args:
            data (DataNode): The row being scored.

        Returns:
            str: The joined context chunks.
        """
        return "\n\n".join(data.context)


faithfulness = Faithfulness()
