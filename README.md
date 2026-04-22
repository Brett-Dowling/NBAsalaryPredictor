# NBA Salary Prediction Using Machine Learning

COSC 325 — Spring 2026 Course Project

## Overview

This project uses supervised machine learning to predict 2025–26 NBA player salaries (in millions of dollars) from performance statistics. We scrape advanced stats, per-game stats, and contract data from Basketball Reference, then train and compare multiple regression models to see which stats the NBA actually pays for.

## Dataset

- **Source:** Basketball Reference (scraped via `playerDataSet.py`)
- **Players:** 438 (after filtering to ≥10 games played)
- **Features:** 28 (24 numeric stats + 4 position dummies)
- **Target:** Salary in $M (log-transformed for modeling)

## Models

| Model | Status | R² |
|-------|--------|----|
| Median Baseline | Done | -0.208 |
| Linear Regression | Done | 0.507 |
| Ridge Regression | Done | 0.520 |
| Decision Tree | Done | 0.622 |
| Random Forest | Done | 0.678 |
| MLP (Neural Network) | Done | 0.500 |

## Repo Structure

- `playerDataSet.py` — Scrapes and cleans data, generates EDA plots
- `baseline_salary_prediction.ipynb` — Preprocessing, models, and evaluation
- `nba_salary_2025_26.csv` — Cleaned dataset (438 players, 29 columns)

## Still To Do

- [x] Decision Tree with overfitting analysis
- [x] Random Forest with feature importance analysis
- [x] MLP (neural network) implementation and tuning
- [ ] Final model comparison table and plots
- [ ] Presentation slides (due Apr 29)
- [ ] Final report — 4-page IEEE format (due May 6)

## How to Run

1. Install dependencies: `pip install pandas numpy scikit-learn matplotlib seaborn`
2. Open `baseline_salary_prediction.ipynb` in VS Code or Jupyter
3. Place `nba_salary_2025_26.csv` in the same directory
4. Run All
