# Sentiment Analysis Agent

A compact command-line sentiment analysis agent for exploring **AI Agents and NLP Foundations**. It uses the lightweight VADER lexicon-and-rule-based NLP library to classify a text as **Positive**, **Negative**, or **Neutral**, and reports its compound sentiment score and positive/neutral/negative proportions.

## Requirements

- Python 3.10 or newer
- pip

## Setup

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
```

## Run

Pass a sentence in quotes:

```bash
python sentiment_agent.py "I love how helpful this tool is!"
```

Or pipe text through standard input:

```bash
echo "The delivery was late and disappointing." | python sentiment_agent.py
```

The agent prints the sentiment label, VADER compound score (from -1 to +1), and the normalized positive, neutral, and negative proportions. VADER's conventional compound thresholds are used: scores at least `0.05` are Positive, scores at most `-0.05` are Negative, and scores between them are Neutral.

## Sample knowledge base

[`sample_data.csv`](sample_data.csv) contains example sentences and their expected sentiment labels. It is a small reference set for trying the CLI; it is not used to train VADER.

## Tests

Install pytest if it is not already available, then run:

```bash
python -m pip install pytest
python -m pytest tests/test_sentiment_agent.py -x -q
```

## How it works

The agent validates that input is non-empty, obtains VADER polarity scores, and applies a transparent threshold-based decision to the compound score. VADER is useful for a small portfolio demo because it is fast, has no model download step, and handles common sentiment cues such as intensifiers, punctuation, and emoticons. It is a rule-based baseline, so context-dependent meaning and domain-specific language can still lead to errors.
