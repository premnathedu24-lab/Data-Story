# Data Story — Project Brief

## Product direction
Data Story is a local-first CSV analytics and machine-learning workbench intended for freelancers preparing client reports and small businesses exploring operational datasets.

## Current capabilities
1. Upload and validate UTF-8 CSV files with comma, semicolon, tab or pipe delimiters.
2. Explore data quality, numeric distributions, categorical distributions and relationships.
3. Train and compare classification and regression model candidates.
4. Run chronological lag-based forecasting with optional date + hour combination and seasonal baselines.
5. Run exploratory clustering with K-Means, Agglomerative and DBSCAN.
6. Inspect metrics, charts, residuals, confusion matrices, thresholds, cluster maps and held-out predictions.
7. Download prediction outputs, forecast outputs, cluster assignments, model artifacts and a standalone HTML report.

## Current limits
- 5 MB CSV maximum.
- 10,000 data rows maximum.
- 30 columns maximum.
- One CSV per session.

These limits remain intentionally unchanged for this release.

## Architecture
- `analysis.py`: CSV validation and deterministic descriptive analysis.
- `enhancements.py`: categorical profiling, encoding advice and chart preparation.
- `ml_engine.py`: supervised learning, forecasting and clustering engines.
- `model_guidance.py`: rule-based project and model guidance.
- `reporting.py`: standalone HTML report generation.
- `app.py`: Streamlit interface.

## Product status
The current version is a local prototype and portfolio project, not production SaaS. It has no authentication, tenant isolation, cloud persistence, payments, billing, hosted deployment, security audit or legal/compliance review. Use synthetic or non-sensitive data until the appropriate controls are designed and tested.

## Validation
The current automated suite contains 101 passing tests in the build environment. Browser-level upload, download, Docker build and hosted deployment still require manual verification.
