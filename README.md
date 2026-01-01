# EASEy

An implementation of [Embarrassingly Shallow Autoencoders (EASE)](https://arxiv.org/abs/1905.03375).

EASE is a state-of-the-art prediction model for collaborative filtering on implicit feedback.

## When to use EASE and when not to use EASE

EASE consistently performs near the top of recommender system benchmarks [(see live benchmark)](https://openbenchmark.github.io/BARS/Matching/leaderboard/amazonbooks_m1.html). It outperforms many more complex (deep learning / graph) approaches [(see paper)](https://arxiv.org/pdf/2203.01155).

EASE is best when the number of items is small, because the most computationally complex part of training is taking the inverse of an item x item cooccurrence matrix. The good news is, this complexity is independent of the number of users or interactions.

EASE also doesn't take into account any item or user features like more complex models - it uses interactions only.

Given these two constraints, EASE is a great tool for:
* Standalone prediction - Raw EASE scores are highly predictive
* Candidate generation - Limit the item space to a set of relevant candidates per user
* Feature engineering - EASE scores can be used in downstream models (e.g., a classification GBM)

## Installation

EASEy depends on `sparse_dot_mkl` and `numpy`. `sparse_dot_mkl` is used for parallel computation of the gram matrix (`X^TX`), because the `scipy` implementation is single-threaded which becomes a bottleneck very quickly.

It is recommended to install `sparse_dot_mkl` with `conda` because this ensures that MKL is linked properly. If you use `conda`, you likely already have MKL installed because `numpy` is built with MKL by default.

## Usage

EASEy is compatible with both `pandas` and `polars` DataFrames. Technically it's compatible with any object that has array-like values accessible with index `[]` syntax, even a basic `dict`. The EASE class has two public methods - `fit` and `predict` - for training and inference respectively.

See `movielens_example.ipynb` for a simple training and inference example.