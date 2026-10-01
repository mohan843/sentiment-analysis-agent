import pytest

from sentiment_agent import SentimentAgent, main


@pytest.fixture
def agent() -> SentimentAgent:
    return SentimentAgent()


def test_classifies_positive_text(agent: SentimentAgent) -> None:
    result = agent.analyze("I absolutely love this wonderful product!")

    assert result.label == "Positive"
    assert result.compound > 0.05
    assert result.positive > result.negative


def test_classifies_negative_text(agent: SentimentAgent) -> None:
    result = agent.analyze("This is a terrible and disappointing experience.")

    assert result.label == "Negative"
    assert result.compound < -0.05
    assert result.negative > result.positive


def test_classifies_neutral_text(agent: SentimentAgent) -> None:
    result = agent.analyze("The package arrived on Tuesday.")

    assert result.label == "Neutral"
    assert -0.05 < result.compound < 0.05


@pytest.mark.parametrize("text", ["", "   ", "\n\t"])
def test_rejects_empty_text(agent: SentimentAgent, text: str) -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        agent.analyze(text)


def test_cli_prints_label_and_scores(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main(["I am happy with the excellent results!"])

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "Sentiment: Positive" in output
    assert "Compound score:" in output
    assert "positive=" in output
