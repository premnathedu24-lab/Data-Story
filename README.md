# Data Story — Local Analytics & ML Workbench

Data Story is a local-first Streamlit application for freelancers and small businesses. Upload a CSV to inspect the data, explore visualizations, train and compare suitable machine-learning models, review held-out predictions, and export a client-facing HTML report.

> **Current status:** feature-rich local prototype / portfolio project. It is not yet a production SaaS. It has no authentication, multi-user isolation, cloud storage, billing, payment processing, or independent security review. Use synthetic or non-sensitive data until those controls exist.

## What it does

### 1. Data understanding
- Dataset dimensions, missing cells, duplicate rows, column types, distinct values and source preview.
- Numeric distributions, categorical bar/pie charts, numeric relationship plots and suitable date-based trends.
- Basic data-quality notes and conservative identifier handling.
- Automatic CSV delimiter detection for comma, semicolon, tab and pipe-separated files.

### 2. Classification
Models compared:
- Majority-class baseline
- Logistic Regression
- Decision Tree
- Random Forest
- Extra Trees
- Linear SVM
- K-Nearest Neighbors
- Histogram Gradient Boosting

Evaluation includes accuracy, balanced accuracy, weighted and macro F1, per-class precision/recall, confusion matrix, ROC-AUC and PR-AUC when probability outputs are available. Binary probability-capable models expose a decision-threshold control, false-positive/false-negative counts, and ROC / Precision–Recall curves.

### 3. Regression
Models compared:
- Mean baseline
- Linear Regression
- Ridge Regression
- Decision Tree Regressor
- Random Forest Regressor
- Extra Trees Regressor
- Histogram Gradient Boosting Regressor

Evaluation includes MAE, RMSE and R². Visuals include actual-vs-predicted with a perfect-prediction reference line, residual distribution, and residual-vs-predicted plot. Repeated maximum target values trigger a possible top-coding/cap warning.

### 4. Forecasting
Models / baselines compared:
- Naive last-value
- Moving average
- Simple exponential smoothing
- Seasonal naive (when a seasonal period is supplied)
- Linear Regression with lag and calendar features
- Ridge Regression with lag and calendar features
- Random Forest with lag and calendar features
- Extra Trees with lag and calendar features

Forecasting uses chronological holdout evaluation, not a random split. It can combine a date column with a separate hour/time column (for example, `dteday` + `hr` in Bike Sharing), derives calendar features, reports inferred observation frequency, and warns about duplicate timestamps aggregated to a mean. It produces held-out predictions and future forecast rows.

**Forecasting scope:** this is a univariate lag-based forecasting engine with derived calendar features. It does not yet train on arbitrary future-known external variables such as future weather or planned promotions. Confirm timestamp frequency, gaps, seasonality and horizon before interpreting results.

### 5. Clustering
- K-Means
- Agglomerative clustering
- DBSCAN
- Median imputation and standard scaling for selected numeric features.
- Silhouette score where a valid partition exists, cluster counts, PCA projection, cluster assignment CSV and model/configuration download.

Clustering is exploratory. A silhouette score is not supervised accuracy and does not prove that clusters are meaningful business segments.

### 6. Model outputs and reports
- Held-out prediction table and CSV download.
- Custom single-row prediction for classification/regression, with class probabilities when available.
- Trained supervised model download (`.pkl`).
- Future forecast CSV and forecasting model/configuration download.
- Cluster assignments CSV and clustering artifact download.
- Standalone HTML report with descriptive profile, chart summaries, model comparison metrics, exclusions and held-out examples.

Pickle files can execute code when loaded. Only load model artifacts from sources you trust, and keep the scikit-learn/Python environment compatible with the environment that created them.

## Models and preprocessing

The tabular pipelines use median imputation for numeric columns, most-frequent imputation and one-hot encoding for categorical columns, and scaling for models that benefit from it. Preprocessing is fitted within each training pipeline. High-cardinality text columns above the configured threshold are excluded with a visible reason; this version does not yet provide TF-IDF or transformer-based text modeling.

Optional 3-fold cross-validation is available for classification/regression. It is slower and does not replace a final holdout or a deployment-matched validation design. Forecasting uses chronological holdout evaluation; rolling-origin backtesting is a future enhancement.

## Run on Windows

Open PowerShell in the extracted project directory:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Open the local address printed by Streamlit, normally `http://127.0.0.1:8501`. Stop the app with `Ctrl+C`.

## Run tests

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -t . -v
```

`test_ui.py` performs static app checks and synthetic ML/report integration. It is not a full browser automation test. Manually verify the Streamlit interface, upload widget, report download and model downloads in your local browser before release.

## Current input limits

- One CSV file per session.
- UTF-8 CSV input.
- Maximum 5 MB.
- Maximum 10,000 data rows.
- Maximum 30 columns.
- Automatic delimiter detection: comma, semicolon, tab or pipe.

These limits are intentionally retained for predictable local resource use. Excel/PDF ingestion, large-data processing, distributed training and multi-user concurrency are not included.

## Important limitations before commercial use

- No login, role-based access, tenant isolation, database, payment gateway or subscription management.
- No hosted deployment or uptime monitoring.
- No production penetration test, privacy/legal review or compliance certification.
- No automated guarantee of fairness, representativeness, causality or model suitability.
- Medical, financial, hiring and other high-impact predictions require domain-specific validation and human oversight; the demo is not a decision authority.
- Metrics are tied to the dataset and validation split. Leakage, repeated entities, time dependence and target definition can make results misleading.
- Feature importance is model-specific association, not causation.

## Main files

- `app.py` — Streamlit interface and charts.
- `analysis.py` — strict CSV validation and descriptive statistics.
- `enhancements.py` — categorical profiling, encoding guidance and chart data.
- `ml_engine.py` — classification, regression, forecasting and clustering engines.
- `model_guidance.py` — rule-based model and project advice.
- `reporting.py` — standalone HTML/SVG report generation.
- `tests/` — automated unit tests and synthetic integration checks.
- `examples/` — optional scripts demonstrating predictions from trusted exported model files.
- `Dockerfile` — container build definition; build/deployment must be verified locally.
