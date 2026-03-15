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

## Dataset

- **Source:** Download the dataset from Kaggle: https://www.kaggle.com/datasets/programmer3/demand-forecasting-dataset
- **How to get it:**
	- Option A (Kaggle website): Go to the URL above, download the CSV, and place it in the `data/` folder.
	- Option B (Kaggle CLI): If you have the Kaggle CLI configured, run:

```bash
kaggle datasets download -d programmer3/demand-forecasting-dataset -p data/ --unzip
```

- **Filename expected by the code:** `data/demand_forecasting_dataset (1).csv` (the repository's `preprocessing.py` reads this path). If your download has a different name, either rename the file or edit `preprocessing.py` to point to the downloaded filename.

## Training

- Create and activate a virtual environment (recommended):

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1   # PowerShell (Windows)
# or .venv\Scripts\activate.bat for cmd
```

- Install dependencies:

```bash
pip install -r requirements.txt
```

- Train the models (this runs the full pipeline: preprocessing, feature engineering, training LSTM and Random Forest, then evaluation):

```bash
python main.py
```

- Output:
	- LSTM model: `models/lstm_model.h5`
	- Random Forest: `models/rf_model.pkl`

- Notes:
	- `main.py` uses `preprocessing.load_data()` to read the CSV, then trains the LSTM (via `lstm_model.train_lstm`) and Random Forest (via `ensemble_model.train_rf`) and combines predictions with `hybrid_model.hybrid_prediction`.
	- To change training hyperparameters (epochs, batch size, RF params), edit `lstm_model.py` and `ensemble_model.py` respectively.
	- For GPU acceleration, install the GPU build of TensorFlow and ensure CUDA/cuDNN are configured.

## Optional: Dashboard

- A simple Streamlit dashboard is provided in `dashboard.py` for quick dataset preview and uploads.
- Install Streamlit and run the dashboard with:

```bash
pip install streamlit
streamlit run dashboard.py
```

## Safer data loading (optional)

- Currently `preprocessing.py` reads a specific filename (`data/demand_forecasting_dataset (1).csv`). If your downloaded CSV has a different name, either rename it to match or update the code.
- A small, robust alternative is to let the script load the first CSV found in `data/`. Example snippet you can use in `preprocessing.py`:

```python
import glob
import pandas as pd

files = glob.glob("data/*.csv")
if not files:
		raise FileNotFoundError("No CSV found in data/ - download dataset and place it there.")

df = pd.read_csv(files[0])
```

This avoids hard-coded filenames and works regardless of the exact downloaded name.

## Front End (what it does and what it shows)

- The repository includes a lightweight front end for quick inspection and visualization:
	- `dashboard.py` — a Streamlit app that allows you to upload a CSV and preview the dataset (shows the first rows in a table).
	- `visualization.py` — plotting utilities used for model inspection and results:
		- `plot_predictions(actual, predicted)`: plots actual vs predicted demand over time (line chart with legend and axes labels).
		- `plot_feature_importance(model, feature_names)`: plots feature importances from tree-based models (bar chart, rotated x labels).

- Usage notes:
	- Run the Streamlit dashboard to upload and preview datasets quickly: `streamlit run dashboard.py`.
	- Use the plotting helpers in `visualization.py` after training to visualize model predictions and feature importances. For example, call `plot_predictions(y_test, final_pred)` or `plot_feature_importance(rf_model, feature_names)` from a notebook or small script.
	- These visualizations use Matplotlib and will open inline in notebooks or in separate windows when run as scripts.

## Examples

- After running `python main.py`, you can visualize results from a Python REPL, script, or notebook. Example usage:

```python
from visualization import plot_predictions, plot_feature_importance
import joblib
import pandas as pd

# If you have saved or computed test labels and final predictions
# (main.py computes `final_pred` and `y_test` during the run), call:
# plot_predictions(y_test, final_pred)

# Example: load trained Random Forest and show feature importance
rf = joblib.load("models/rf_model.pkl")
feature_names = [
		'product_id','category_id','store_id','price','promotion_flag',
		'holiday_flag','economic_index','day','month','year','dayofweek'
]
plot_feature_importance(rf, feature_names)
```

Notes:
- If you run these from a script on Windows, Matplotlib will open a new window for the plots. In Jupyter notebooks, plots render inline.

## Optional frontend dependencies

- For the dashboard and improved plotting you may want to install:

```bash
pip install streamlit seaborn
```

- Add these to `requirements.txt` if you want them recorded as project dependencies.

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
