# Data Story — Release checklist

## Implemented in this build
- [x] Automatic comma/semicolon/tab/pipe delimiter detection.
- [x] Colored Streamlit theme and consistent chart palette.
- [x] Classification model expansion, class metrics, threshold controls and ROC/PR plots.
- [x] Regression model expansion, perfect-prediction line and residual diagnostics.
- [x] Forecasting model expansion, chronological split, date + hour support and calendar features.
- [x] K-Means, Agglomerative and DBSCAN clustering workflow.
- [x] Explain why selected features are excluded.
- [x] Prediction, forecast, clustering and model artifact downloads.
- [x] Optional 3-fold cross-validation for tabular classification/regression.
- [x] 101 automated tests passing in the build environment.

## Before public release
- [ ] Run the updated app locally in a browser.
- [ ] Test real CSV uploads with comma and semicolon delimiters.
- [ ] Test all four workflows: classification, regression, forecasting and clustering.
- [ ] Verify all download buttons and open exported files.
- [ ] Verify hourly Bike Sharing data using both date and hour columns.
- [ ] Build Docker image and verify container startup.
- [ ] Create GitHub repository and verify GitHub Actions.
- [ ] Add screenshots and a short demo video to the repository.
- [ ] Review privacy/security requirements before accepting real client data.

## Explicitly not included
- Authentication, tenant isolation, billing/payments or cloud storage.
- Hosted production deployment or uptime monitoring.
- Arbitrary high-cardinality text modeling / transformer embeddings.
- Multivariate forecasting with future exogenous variables.
- Rolling-origin forecasting backtesting.
- XGBoost dependency; it can be evaluated as an optional future extension.
