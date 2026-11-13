# Customer Churn Prediction & Retention Scoring

A machine learning project that estimates telecom customer churn risk, explains model predictions with SHAP, and presents retention priorities in Streamlit.

**Python · scikit-learn · XGBoost · LightGBM · SHAP · Streamlit**

[Try the Streamlit demo](https://telco-retention-ai.streamlit.app)

## What it does

- Compares Logistic Regression, XGBoost, and LightGBM.
- Adds customer lifecycle, service, contract, and payment features.
- Produces churn probabilities and HIGH / MEDIUM / LOW risk groups.
- Generates SHAP explanations and rule-based retention suggestions for high-risk customers.
- Provides dashboard pages for overview, model performance, risk segmentation, SHAP, recommendations, and customer drill-down.

### Dataset preview

![Churn distribution from the repository's saved analysis](outputs/churn_distribution.png)

## Run locally

Use Python 3.10 or newer. The bundled IBM Telco CSV is at [data/WA_Fn-UseC_-Telco-Customer-Churn.csv](data/WA_Fn-UseC_-Telco-Customer-Churn.csv).

```bash
git clone https://github.com/gulshairdev/customer-churn-prediction-system.git
cd customer-churn-prediction-system
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe main.py
.venv\Scripts\python.exe -m streamlit run app.py
```

macOS / Linux:

```bash
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python main.py
.venv/bin/python -m streamlit run app.py
```

Run commands from the repository root. The pipeline writes models, charts, scoring CSVs, and recommendations into `models/` and `outputs/`. Streamlit prints the local dashboard URL, normally `http://localhost:8501`.

The repository includes artifacts from an earlier run, so you can also launch the dashboard directly. Retrain with `main.py` if those files are missing or incompatible with your installed package versions. The dashboard reads saved models and outputs; it does not retrain the models itself.

## Data and workflow

| Item | Current implementation |
| --- | --- |
| Raw dataset | 7,043 customers; 5,174 stayed and 1,869 churned |
| Target | `Churn`: Yes → 1, No → 0 |
| Engineered features | Tenure group, charge ratio, service count, high-value flag, contract/payment risk, missing support services |
| Modeling features | 32 after the current engineering and encoding steps |
| Split | 5,634 training / 1,409 holdout rows; stratified, `random_state=42` |
| Model selection | Highest holdout AUC-ROC among the three candidates |

`main.py` loads the data, engineers features, preprocesses, builds EDA charts, compares models, generates SHAP explanations, scores a demonstration batch, creates recommendations, and saves artifacts.

## Reproduced baseline

The Logistic Regression baseline was reproduced using the current data loading, feature engineering, preprocessing, and `class_weight="balanced"` settings:

| Metric | Baseline score |
| --- | ---: |
| AUC-ROC | 0.8447 |
| Precision | 0.5130 |
| Recall | 0.7888 |
| F1 | 0.6217 |
| Accuracy | 0.7452 |

[Verification record and package versions](docs/verification.json)

This verification retrained the Logistic Regression baseline. Run `main.py` to reproduce the full three-model comparison and SHAP outputs. Saved plots may reflect a different package environment or an earlier run.

The current implementation selects the model on the same holdout used for reporting, and some preprocessing is performed before splitting. A separate validation/test design and a training-only preprocessing pipeline are the next steps for an independent final evaluation.

## Risk groups and outputs

| Probability | Risk group | Example follow-up |
| --- | --- | --- |
| `p >= 0.70` | HIGH | Review the top churn drivers and contact the customer |
| `0.40 <= p < 0.70` | MEDIUM | Add to a monitoring or nurture list |
| `p < 0.40` | LOW | Routine engagement |

The current demo scores the held-out batch and writes:

- `outputs/weekly_scoring_week_1.csv`: probability, risk level, batch index, week, and date.
- `outputs/retention_recommendations.csv`: high-risk customers, their top drivers, and suggested actions.
- `models/best_model.pkl`, `models/scaler.pkl`, `models/feature_names.pkl`, and `models/results_dict.pkl`.

The workflow is run manually; no weekly scheduler is included. `customer_index` refers to a position in the scored batch, not the original customer ID. Retention actions are rules for demonstration and need review before business use.

## Repository guide

| Path | Purpose |
| --- | --- |
| [app.py](app.py) | Streamlit dashboard |
| [main.py](main.py) | Pipeline orchestration |
| [src/data_loader.py](src/data_loader.py) | Loading and data inspection |
| [src/feature_engineer.py](src/feature_engineer.py) | Customer features |
| [src/preprocessor.py](src/preprocessor.py) | Encoding, split, and scaling |
| [src/model_trainer.py](src/model_trainer.py) | Three-model training and selection |
| [src/explainer.py](src/explainer.py) | SHAP explanations |
| [src/scorer.py](src/scorer.py) | Batch scoring and risk groups |
| [src/recommender.py](src/recommender.py) | Rule-based retention suggestions |
| [src/visualizer.py](src/visualizer.py) | Analysis and model-result charts |

## Next steps

Improve evaluation isolation, preserve original customer IDs through scoring, compare probability calibration and decision thresholds, and add automated checks for training and inference consistency.

**Author:** [Gul Shair](https://github.com/gulshairdev) · Teyzix Core ML-3 internship project
