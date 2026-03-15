# MINI_PROJECT_UPDATED

Small demand-forecasting project containing preprocessing, feature engineering, and LSTM/hybrid/ensemble models.

## Requirements
- Python 3.8+
- pandas, numpy, scikit-learn, matplotlib, tensorflow (or keras)

Install common dependencies:

```bash
python -m pip install pandas numpy scikit-learn matplotlib tensorflow
```

## Run
Start the main script:

```bash
python main.py
```

## Repo structure (key files)
- `main.py` — entry point
- `preprocessing.py` — data cleaning
- `feature_engineering.py` — feature creation
- `lstm_model.py` — LSTM model code
- `hybrid_model.py`, `ensemble_model.py` — combined approaches
- `evaluation.py` — evaluation helpers
- `supply_chain.py` — domain-specific utilities
- `data/` — dataset (demand_forecasting_dataset (1).csv)
- `models/` — saved models (lstm_model.h5)

## Notes
- Adjust dependency installs if you use a GPU build of TensorFlow.
- If you prefer, create a `requirements.txt` for exact versions.
