"""A small rule-based sentiment analysis agent powered by VADER."""

import argparse
import sys
from dataclasses import dataclass

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


@dataclass(frozen=True)
class SentimentResult:
    label: str
    compound: float
    positive: float
    neutral: float
    negative: float


class SentimentAgent:
    """Analyze text and classify its overall sentiment."""

    def __init__(self) -> None:
        self._analyzer = SentimentIntensityAnalyzer()

    def analyze(self, text: str) -> SentimentResult:
        """Return VADER scores and a label for non-empty text."""
        if not text or not text.strip():
            raise ValueError("Text must not be empty.")

        scores = self._analyzer.polarity_scores(text)
        compound = scores["compound"]
        if compound >= 0.05:
            label = "Positive"
        elif compound <= -0.05:
            label = "Negative"
        else:
            label = "Neutral"

        return SentimentResult(
            label=label,
            compound=compound,
            positive=scores["pos"],
            neutral=scores["neu"],
            negative=scores["neg"],
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Classify text as Positive, Negative, or Neutral using VADER."
    )
    parser.add_argument(
        "text",
        nargs="*",
        help="Text to analyze. If omitted, text is read from standard input.",
    )
    args = parser.parse_args(argv)

    text = " ".join(args.text) if args.text else sys.stdin.read().strip()
    if not text:
        parser.error("provide text as an argument or through standard input")

    result = SentimentAgent().analyze(text)
    print(f"Sentiment: {result.label}")
    print(f"Compound score: {result.compound:+.3f}")
    print(
        "Scores: "
        f"positive={result.positive:.3f}, "
        f"neutral={result.neutral:.3f}, "
        f"negative={result.negative:.3f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
