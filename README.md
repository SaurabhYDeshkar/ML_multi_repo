# ML Refresh & Reference

A collection of personal notes, reference snippets, and end-to-end Machine Learning projects used to refresh concepts and APIs without repeatedly searching documentation.

The repository has two purposes:

* **Old Reference** – Quick reminders for common preprocessing steps, estimators, GridSearchCV usage, ensembling techniques, time-series methods, and other frequently used APIs.
* **Projects** – Complete, reusable ML pipelines covering the major learning paradigms used in practice.

## Repository Structure

```text
old_reference/
    Personal API reminders and coursework snippets.

classification/
    End-to-end classification pipeline.

regression/
    End-to-end regression pipeline.

time_series/
    Forecasting and temporal modelling pipeline.

deep_learning/
    Neural network based pipeline.

common/
    Shared utilities and reusable components.
```

## Typical Machine Learning Workflow

A typical production-oriented ML workflow consists of:

1. Problem definition
2. Data loading
3. Exploratory Data Analysis (EDA)
4. Data preprocessing
5. Feature engineering
6. Train/validation/test split
7. Model selection
8. Hyperparameter tuning
9. Model training
10. Model evaluation
11. Model comparison
12. Model serialization
13. Inference
14. Deployment
15. Monitoring and retraining

Not every project requires every stage, but this serves as the general blueprint followed throughout this repository.

## Goals

This repository focuses on reinforcing concepts rather than memorizing individual APIs.

Each project is intended to answer questions such as:

* Why was this preprocessing step chosen?
* Why is this model appropriate?
* Which metrics best evaluate it?
* How can the pipeline be made modular and reusable?
* How can the trained model be packaged for inference?

The `old_reference/` directory contains concise reminders, while the project directories contain complete implementations built around those ideas.