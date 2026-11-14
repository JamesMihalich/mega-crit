# Mega-Crit

This repo is a sibling of the broader `Mega-Seg` family, but stands on it's own as an important tool for training and quantifying the quality of machine learning networks. `Mega-Crit` provides a library of ever-growing criterions (hence the name) that can stand alone, or be initialized by a unified `Criterion` API.

## Design Philosophy

We divide criterion into two distinct groups. Losses is the first, representing differentiable functions. These are capable loss functions useful for model training. Metrics is the second, representing non-differentiable quality metrics important for post-training model analysis.

## Basic Use
