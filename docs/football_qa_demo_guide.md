# Football NLQA Demo Guide

This guide explains how to run and test the integrated Football Natural
Language Question Answering application.

## Open the Demo

The application can run directly in Google Colab without Google Drive access
or manual model downloads.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/eshan14git/football-qa-nlp/blob/disath-dev/notebooks/football_qa_demo.ipynb)

## Running Instructions

1. Click the **Open in Colab** button above.
2. Select **Runtime → Run all**.
3. Wait for the repository, models and compressed dataset to load.
4. Open the Gradio link displayed near the bottom of the notebook.
5. Enter a football question or select one of the provided examples.

No Google Drive permission is required.

## Supported Questions

The application supports:

- Match winner
- Match score
- Match goal scorers
- A player's goal count in a match
- A player's scoring minutes in a match

## Example

Question:

```text
Who won between Brazil and Germany on 8 July 2014?
```

Expected answer:

```text
Germany won against Brazil 7-1 on 2014-07-08.
```

## Models

The interface includes:

- Word-and-character TF-IDF Logistic Regression
- Word-level one-dimensional CNN

| Model | Generated-Test Accuracy | Manual Accuracy | Manual Macro F1 |
|---|---:|---:|---:|
| Logistic Regression | 100% | 74% | 0.7381 |
| 1D CNN | 100% | 66% | 0.6571 |

Logistic Regression is used as the final retrieval model because it achieved
better performance on the independent manual robustness test. The CNN remains
available for comparison.

## Runtime Files

The required trained artifacts are stored in:

```text
models/disath/
```

The compressed factual dataset is stored at:

```text
data/runtime/football_qa_v2_master.csv.gz
```

The notebook automatically clones the required branch and loads these files.

## Branch Note

The current Colab link runs the `disath-dev` branch. If this implementation is
merged into the final `main` branch, update the Colab link from:

```text
blob/disath-dev/
```

to:

```text
blob/main/
```
