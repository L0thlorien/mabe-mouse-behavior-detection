# MABe Mouse Behavior Detection - Modular Baseline

A clean, modular baseline for the MABe Mouse Behavior Detection competition.

Entered the competition too late, didn't have time to submit

## Quick Start

```bash
uv run train.py

uv run predict.py
```

## Structure

```
src/
├── config.py           # All settings in one place
├── data_loader.py      # Load CSV + Parquet
├── features.py         # Feature engineering
├── models.py           # XGBoost/LightGBM/CatBoost
├── postprocessing.py   # Predictions → Segments
└── pipeline.py         # Orchestration
```

