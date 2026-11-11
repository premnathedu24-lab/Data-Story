"""Local end-to-end ML engines for Data Story. No network calls."""

from __future__ import annotations

import io

import csv

import re

import warnings

import numpy as np

import pandas as pd

from sklearn.compose import ColumnTransformer

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sklearn.pipeline import Pipeline

from sklearn.dummy import DummyClassifier, DummyRegressor

from sklearn.linear_model import LogisticRegression, Ridge, LinearRegression

from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from sklearn.ensemble import (RandomForestClassifier, RandomForestRegressor,

    ExtraTreesClassifier, ExtraTreesRegressor, HistGradientBoostingClassifier, HistGradientBoostingRegressor)

from sklearn.svm import LinearSVC

from sklearn.calibration import CalibratedClassifierCV

from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import train_test_split, StratifiedKFold, KFold, cross_validate

from sklearn.metrics import (accuracy_score, balanced_accuracy_score,

    precision_recall_fscore_support, confusion_matrix, mean_absolute_error,

    mean_squared_error, r2_score, classification_report, f1_score,

    roc_auc_score, average_precision_score, silhouette_score)

from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN

from sklearn.utils.multiclass import type_of_target





def _is_identifier(name: str) -> bool:

    raw = str(name).strip()

    normalized = re.sub(r'[^a-z0-9]+', '', raw.lower())

    exact = {'id','uuid','email','phone','name','address','account','postcode',

             'zipcode','token','passengerid','customerid','transactionid',

             'recordid','rowid','userid','orderid','productid','memberid','unnamed0','index','rownumber','instant'}

    camel_id = bool(re.search(r'(?:^|[A-Za-z])Id$', raw))

    suffix = bool(re.search(r'(?:^|[_\s-])(?:id|uuid|email|phone|token)$', raw, re.I))

    return normalized in exact or camel_id or suffix or any(

        word in normalized for word in ('email','phone','address','account','postcode','zipcode'))







def _looks_like_sequential_index(series: pd.Series) -> bool:

    if not pd.api.types.is_numeric_dtype(series):

        return False

    values = pd.to_numeric(series, errors='coerce').dropna().to_numpy(dtype=float)

    if len(values) < 5 or len(np.unique(values)) != len(values):

        return False

    if not np.all(np.isfinite(values)) or not np.all(np.equal(values, np.floor(values))):

        return False

    ordered = np.sort(values)

    diffs = np.diff(ordered)

    return bool(len(diffs) and np.all(diffs == diffs[0]) and diffs[0] > 0)





def _date_ratio(series: pd.Series) -> float:

    if pd.api.types.is_numeric_dtype(series):

        return 0.0

    with warnings.catch_warnings():

        warnings.simplefilter('ignore', UserWarning)

        return float(pd.to_datetime(series, errors='coerce', utc=True).notna().mean())





def _numeric_series(s: pd.Series) -> pd.Series:

    if pd.api.types.is_numeric_dtype(s):

        return pd.to_numeric(s, errors='coerce')

    cleaned = (s.astype('string').str.strip().str.replace(',', '', regex=False)

               .str.replace('$', '', regex=False).str.replace('₹', '', regex=False)

               .str.replace('%', '', regex=False))

    return pd.to_numeric(cleaned, errors='coerce')





def _detect_delimiter(text: str) -> str:

    sample = text[:65536]

    try:

        return csv.Sniffer().sniff(sample, delimiters=',;\t|').delimiter

    except Exception:

        # A stable fallback for ordinary CSVs and small/ambiguous samples.

        first = next((line for line in sample.splitlines() if line.strip()), '')

        counts = {d: first.count(d) for d in (',', ';', '\t', '|')}

        return max(counts, key=counts.get) if max(counts.values(), default=0) else ','





def load_frame(data: bytes) -> pd.DataFrame:

    import csv

    if not data:

        raise ValueError('The uploaded file is empty.')

    text = data.decode('utf-8-sig')

    delimiter = _detect_delimiter(text)

    df = pd.read_csv(io.StringIO(text), sep=delimiter)

    df.columns = [str(c).strip() for c in df.columns]

    for c in df.columns:

        if df[c].dtype == object:

            converted = _numeric_series(df[c])

            nonblank = df[c].notna() & df[c].astype(str).str.strip().ne('')

            if nonblank.sum() and converted[nonblank].notna().mean() >= .95:

                df[c] = converted

    return df





def infer_task(y: pd.Series) -> str:

    if pd.api.types.is_numeric_dtype(y):

        uniq = y.nunique(dropna=True)

        if uniq <= 15 and uniq <= max(2, int(len(y.dropna()) * .05)):

            return 'classification'

        return 'regression'

    return 'classification'





def _feature_types(X):

    numeric = X.select_dtypes(include=np.number).columns.tolist()

    categorical = [c for c in X.columns if c not in numeric]

    return numeric, categorical





def _pipeline(X, model, scale=False, dense=False):

    numeric, categorical = _feature_types(X)

    transformers = []

    if numeric:

        steps = [('imputer', SimpleImputer(strategy='median'))]

        if scale:

            steps.append(('scaler', StandardScaler()))

        transformers.append(('numeric', Pipeline(steps), numeric))

    if categorical:

        transformers.append(('categorical', Pipeline([

            ('imputer', SimpleImputer(strategy='most_frequent')),

            ('onehot', OneHotEncoder(handle_unknown='ignore', min_frequency=2, sparse_output=not dense))

        ]), categorical))

    prep = ColumnTransformer(transformers, remainder='drop')

    return Pipeline([('preprocess', prep), ('model', model)])





def _feature_selection(df, target, features):

    excluded = []

    chosen = list(features) if features is not None else [c for c in df.columns if c != target]

    chosen = [c for c in chosen if c in df.columns and c != target]

    kept = []

    for col in chosen:

        if _is_identifier(col):

            excluded.append({'feature': col, 'reason': 'Likely identifier; excluded to avoid learning row/entity IDs.'})

        elif _looks_like_sequential_index(df[col]):

            excluded.append({'feature': col, 'reason': 'Values form a unique sequential index; excluded to avoid learning row order.'})

        elif _date_ratio(df[col]) >= .8:

            excluded.append({'feature': col, 'reason': 'Date/time column; use it as a time axis or derive justified calendar features.'})

        elif df[col].nunique(dropna=True) <= 1:

            excluded.append({'feature': col, 'reason': 'Constant or all-missing column; it contains no usable variation.'})

        elif not pd.api.types.is_numeric_dtype(df[col]) and df[col].nunique(dropna=True) > 100:

            excluded.append({'feature': col, 'reason': 'High-cardinality text/category (>100 unique values); text encoding is not enabled in this tabular pipeline.'})

        else:

            kept.append(col)

    # Known deterministic leakage pattern in the Bike Sharing dataset.

    if str(target).strip().lower() == 'cnt':

        for col in ('casual', 'registered'):

            if col in kept:

                kept.remove(col)

                excluded.append({'feature': col, 'reason': 'Target leakage: Bike Sharing cnt is the sum of casual and registered.'})

    return kept, excluded





def _cv_for_models(models, X, y, task, random_state):

    """Optional 3-fold CV; return mean/std for a task-appropriate metric."""

    if task == 'classification':

        counts = pd.Series(y).value_counts()

        if counts.min() < 3:

            return {}, 'Cross-validation skipped: the smallest class has fewer than 3 rows.'

        splitter = StratifiedKFold(n_splits=3, shuffle=True, random_state=random_state)

        scoring = {'primary': 'f1_weighted'}

    else:

        if len(y) < 12:

            return {}, 'Cross-validation skipped: fewer than 12 usable rows.'

        splitter = KFold(n_splits=3, shuffle=True, random_state=random_state)

        scoring = {'primary': 'neg_root_mean_squared_error'}

    out = {}

    for name, model in models.items():

        pipe = _pipeline(X, model, scale=('Logistic' in name or 'Ridge' in name or 'SVM' in name or 'KNN' in name), dense=('Gradient Boosting' in name))

        try:

            with warnings.catch_warnings():

                warnings.simplefilter('ignore')

                scores = cross_validate(pipe, X, y, cv=splitter, scoring=scoring,

                                        n_jobs=1, error_score='raise')['test_primary']

            values = scores if task == 'classification' else -scores

            out[name] = {'mean': float(np.mean(values)), 'std': float(np.std(values))}

        except Exception as exc:

            out[name] = {'mean': None, 'std': None, 'error': str(exc)[:180]}

    return out, '3-fold stratified CV' if task == 'classification' else '3-fold shuffled CV'





def train_model(data: bytes, target: str, task: str='Auto', features=None,

                test_size: float=.2, random_state: int=42, run_cv: bool=False):

    df = load_frame(data)

    if target not in df:

        raise ValueError('Target column is not in the uploaded data.')

    if _is_identifier(target):

        raise ValueError('The selected target looks like an identifier. Choose a meaningful outcome column instead of an ID/index.')

    y_raw = df[target]

    valid = y_raw.notna()

    if not pd.api.types.is_numeric_dtype(y_raw):

        valid &= y_raw.astype('string').str.strip().ne('').fillna(False)

    df = df.loc[valid].copy(); y_raw = y_raw.loc[valid]

    if len(df) < 20:

        raise ValueError('At least 20 rows with a nonblank target are required for a basic train/test evaluation.')

    if task == 'Auto':

        task = infer_task(y_raw)

    if task not in ('classification', 'regression'):

        raise ValueError('Choose Classification or Regression. Forecasting and Clustering are handled separately.')

    target_warnings = []

    if task == 'regression':

        y = _numeric_series(y_raw)

        keep = y.notna() & np.isfinite(y)

        df = df.loc[keep].copy(); y = y.loc[keep].astype(float)

        if y.nunique() < 2:

            raise ValueError('Regression needs at least two distinct numeric target values.')

        max_value = float(y.max())

        max_count = int((y == max_value).sum())

        if max_count >= max(5, int(len(y)*.02)):

            target_warnings.append(f'The maximum target value ({max_value:g}) repeats in {max_count} rows ({max_count/len(y):.1%}). Check whether the source records are capped/top-coded before interpreting predictions near this ceiling.')

    else:

        y = y_raw.astype('string').str.strip().astype(str)

        if y.nunique() < 2:

            raise ValueError('Classification needs at least two target classes.')

        if y.nunique() > 30 or y.nunique() >= len(y) * .5:

            raise ValueError('This target has too many unique labels. It may be an ID or free text; choose a meaningful class column.')

        if y.value_counts().min() < 2:

            raise ValueError('Every class needs at least two examples for a reliable train/test split.')

    chosen, excluded = _feature_selection(df, target, features)

    if not chosen:

        raise ValueError('No usable input features remain. Select meaningful columns and exclude IDs.')

    X = df[chosen].copy()

    for col in X.columns:

        if not pd.api.types.is_numeric_dtype(X[col]):

            values = X[col].astype('string').str.strip().replace('', pd.NA)

            X[col] = values.astype(object).where(values.notna(), np.nan)

    if task == 'classification':

        counts = y.value_counts(); n_test = int(np.ceil(len(y) * test_size))

        strat = y if counts.min() >= 2 and n_test >= y.nunique() and (len(y)-n_test) >= y.nunique() else None

        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size,

            random_state=random_state, stratify=strat)

        models = {

            'Baseline (majority class)': DummyClassifier(strategy='most_frequent'),

            'Logistic Regression': LogisticRegression(max_iter=2000, class_weight=None),

            'Decision Tree': DecisionTreeClassifier(min_samples_leaf=2, class_weight=None, random_state=random_state),

            'Random Forest': RandomForestClassifier(n_estimators=200, min_samples_leaf=2, class_weight=None, random_state=random_state, n_jobs=-1),

            'Extra Trees': ExtraTreesClassifier(n_estimators=200, min_samples_leaf=2, class_weight=None, random_state=random_state, n_jobs=-1),

            'Linear SVM': CalibratedClassifierCV(LinearSVC(class_weight=None, random_state=random_state, max_iter=5000), method='sigmoid', cv=3),

            'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=min(5, max(1, len(Xtr)//10)), weights='distance', n_jobs=-1),

            'Gradient Boosting': HistGradientBoostingClassifier(max_iter=120, learning_rate=.08, l2_regularization=1.0, random_state=random_state),

        }

    else:

        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size, random_state=random_state)

        models = {

            'Baseline (mean)': DummyRegressor(strategy='mean'),

            'Linear Regression': LinearRegression(),

            'Ridge Regression': Ridge(alpha=1.0),

            'Decision Tree': DecisionTreeRegressor(min_samples_leaf=2, random_state=random_state),

            'Random Forest': RandomForestRegressor(n_estimators=200, min_samples_leaf=2, random_state=random_state, n_jobs=-1),

            'Extra Trees': ExtraTreesRegressor(n_estimators=200, min_samples_leaf=2, random_state=random_state, n_jobs=-1),

            'Gradient Boosting': HistGradientBoostingRegressor(max_iter=120, learning_rate=.08, l2_regularization=1.0, random_state=random_state),

        }

    cv_results, cv_method = ({}, 'Not run')

    if run_cv:

        cv_results, cv_method = _cv_for_models(models, Xtr, ytr, task, random_state)

    results = {}; fitted = {}; predictions = {}; probability_map = {}

    for name, model in models.items():

        scale = ('Logistic' in name or 'Ridge' in name or 'SVM' in name or 'K-Nearest' in name)

        pipe = _pipeline(Xtr, model, scale=scale, dense=('Gradient Boosting' in name))

        pipe.fit(Xtr, ytr); pred = pipe.predict(Xte)

        fitted[name] = pipe; predictions[name] = pred

        if task == 'classification':

            pr, rc, f1, _ = precision_recall_fscore_support(yte, pred, average='weighted', zero_division=0)

            macro_f1 = f1_score(yte, pred, average='macro', zero_division=0)

            values = {'Accuracy': float(accuracy_score(yte, pred)),

                      'Balanced accuracy': float(balanced_accuracy_score(yte, pred)),

                      'Precision (weighted)': float(pr), 'Recall (weighted)': float(rc),

                      'F1 (weighted)': float(f1), 'F1 (macro)': float(macro_f1)}

            if hasattr(pipe, 'predict_proba'):

                try:

                    proba = pipe.predict_proba(Xte)

                    probability_map[name] = proba

                    classes = pipe.named_steps['model'].classes_

                    if len(classes) == 2:

                        values['ROC-AUC'] = float(roc_auc_score(yte, proba[:, 1]))

                        values['PR-AUC'] = float(average_precision_score((np.asarray(yte)==classes[1]).astype(int), proba[:, 1]))

                    else:

                        values['ROC-AUC (OvR macro)'] = float(roc_auc_score(yte, proba, multi_class='ovr', average='macro', labels=classes))

                        ap_scores = [average_precision_score((np.asarray(yte)==label).astype(int), proba[:,i]) for i,label in enumerate(classes)]

                        values['PR-AUC (OvR macro)'] = float(np.mean(ap_scores))

                except Exception:

                    pass

        else:

            rmse = float(np.sqrt(mean_squared_error(yte, pred)))

            values = {'MAE': float(mean_absolute_error(yte, pred)),

                      'RMSE': rmse, 'R²': float(r2_score(yte, pred))}

        results[name] = values

    primary_metric = 'F1 (weighted)' if task == 'classification' else 'RMSE'

    best = (max(results, key=lambda k: results[k][primary_metric]) if task == 'classification'

            else min(results, key=lambda k: results[k][primary_metric]))

    best_pred = predictions[best]

    out = pd.DataFrame({'Actual': np.asarray(yte), 'Predicted': best_pred}, index=Xte.index)

    if task == 'classification':

        labels = sorted(pd.Series(y).unique().tolist())

        cm_df = pd.DataFrame(confusion_matrix(yte, best_pred, labels=labels), index=labels, columns=labels)

        report = classification_report(yte, best_pred, output_dict=True, zero_division=0)

    else:

        cm_df = None; report = None

    importance = []

    estimator = fitted[best].named_steps['model']

    if hasattr(estimator, 'feature_importances_') or hasattr(estimator, 'coef_'):

        try:

            names = fitted[best].named_steps['preprocess'].get_feature_names_out()

            vals = (estimator.feature_importances_ if hasattr(estimator, 'feature_importances_')

                    else np.mean(np.abs(estimator.coef_), axis=0) if np.asarray(estimator.coef_).ndim > 1 else np.abs(estimator.coef_))

            importance = sorted([{'feature': str(n).replace('numeric__','').replace('categorical__',''), 'importance': float(v)}

                                 for n,v in zip(names, vals)], key=lambda z: -z['importance'])[:20]

        except Exception:

            pass

    return {'task':task,'target':target,'features':list(X.columns),'excluded_features':excluded,

        'rows_used':int(len(X)),'train_rows':len(Xtr),'test_rows':len(Xte),

        'results':results,'best_model':best,'best_metrics':results[best],

        'predictions':out,'confusion_matrix':cm_df,'classification_report':report,

        'probabilities':probability_map.get(best),'classes':list(getattr(estimator,'classes_',[])),

        'feature_importance':importance,'fitted_models':fitted,'best_pipeline':fitted[best],

        'test_indices':list(Xte.index),'X_test':Xte,'y_test':yte,

        'cross_validation':cv_results,'cross_validation_method':cv_method,

        'target_warnings':target_warnings,'selection_metric':primary_metric}





def _forecast_timestamp(df, date_column, time_column=None):

    if time_column and time_column in df.columns:

        date_text = df[date_column].astype('string').str.strip()

        time_text = df[time_column].astype('string').str.strip()

        # Numeric hour columns such as Bike Sharing's `hr` are zero-padded.

        numeric_hour = pd.to_numeric(df[time_column], errors='coerce')

        if numeric_hour.notna().mean() >= .8 and numeric_hour.dropna().between(0, 23).all():

            time_text = numeric_hour.round().astype('Int64').astype('string').str.zfill(2) + ':00:00'

        combined = date_text + ' ' + time_text

        return pd.to_datetime(combined, errors='coerce', utc=True)

    return pd.to_datetime(df[date_column], errors='coerce', utc=True)





def _calendar_features(dates):

    ix = pd.DatetimeIndex(dates)

    out = pd.DataFrame(index=np.arange(len(ix)))

    out['calendar_hour'] = ix.hour

    out['calendar_dayofweek'] = ix.dayofweek

    out['calendar_month'] = ix.month

    out['calendar_dayofyear'] = ix.dayofyear

    out['calendar_weekend'] = (ix.dayofweek >= 5).astype(int)

    return out





def train_forecast(data: bytes, target: str, date_column: str, lags: int = 5,
                   horizon: int = 1, random_state: int = 42,
                   time_column: str | None = None, seasonal_period: int = 0):
    """Univariate chronological forecast with calendar features and seasonal baselines.

    The model predicts exactly one value at the selected forecast horizon.
    Lags describe the observed history available at the forecast origin.
    """
    df = load_frame(data)
    if target not in df or date_column not in df:
        raise ValueError('Select valid date and target columns.')
    if _is_identifier(target):
        raise ValueError('The forecasting target looks like an identifier. Choose a real measured value to forecast.')
    if not isinstance(lags, int) or lags < 1:
        raise ValueError('Lags must be a positive integer.')
    if not isinstance(horizon, int) or horizon < 1:
        raise ValueError('Forecast horizon must be a positive integer.')
    if not isinstance(seasonal_period, int) or seasonal_period < 0:
        raise ValueError('Seasonal period must be zero or a positive integer.')

    dates = _forecast_timestamp(df, date_column, time_column)
    values = _numeric_series(df[target])
    work = pd.DataFrame({'date': dates, 'value': values}).dropna().sort_values('date')
    duplicate_count = int(work['date'].duplicated().sum())
    if work['date'].duplicated().any():
        work = work.groupby('date', as_index=False)['value'].mean().sort_values('date')
    if len(work) < max(30, lags * 4):
        raise ValueError(f'Forecasting needs at least {max(30, lags * 4)} valid ordered observations; found {len(work)}.')

    work = work.reset_index(drop=True)
    full_dates = pd.DatetimeIndex(work['date'])
    full_history = work['value'].to_numpy(dtype=float)
    last_observed_date = full_dates.max()

    # Infer the observed step size. The same interval is used for target dates
    # during training and for the future forecast timestamp.
    delta = full_dates.to_series().diff().dropna().median()
    if pd.isna(delta) or delta <= pd.Timedelta(0):
        raise ValueError('Could not infer a valid time frequency from the date column.')

    target_dates = full_dates + delta * horizon
    work['target_date'] = target_dates

    # At forecast origin t, lag_1 must be y(t), lag_2 must be y(t-1), etc.
    # This keeps training features consistent with the features built at inference.
    for k in range(1, lags + 1):
        work[f'lag_{k}'] = work['value'].shift(k - 1)
    work['y_future'] = work['value'].shift(-horizon)

    # For a seasonal period p and horizon h, the matching seasonal value is
    # y(t + h - p), i.e. shift p-h observations from the forecast origin.
    use_seasonal_lag = bool(seasonal_period and seasonal_period >= horizon)
    if use_seasonal_lag:
        work['seasonal_lag'] = work['value'].shift(seasonal_period - horizon)

    cal = _calendar_features(target_dates)
    for column in cal:
        work[column] = cal[column].to_numpy()

    work = work.dropna().reset_index(drop=True)
    if len(work) < 20:
        raise ValueError('Not enough rows remain after creating lag features.')

    split = max(1, int(len(work) * 0.8))
    train = work.iloc[:split]
    test = work.iloc[split:]
    if len(test) < 3:
        raise ValueError('Not enough chronological test observations; use more history.')

    feature_cols = [f'lag_{k}' for k in range(1, lags + 1)] + list(cal.columns)
    if use_seasonal_lag:
        feature_cols.append('seasonal_lag')

    Xtr, Xte = train[feature_cols], test[feature_cols]
    ytr, yte = train['y_future'], test['y_future']

    from sklearn.linear_model import LinearRegression
    models = {
        'Naive (last value)': None,
        'Moving Average (recent lags)': 'moving_average',
        'Exponential Smoothing': 'exp_smoothing',
        'Seasonal Naive': 'seasonal_naive' if use_seasonal_lag and seasonal_period > 1 else None,
        'Linear Regression (lags + calendar)': LinearRegression(),
        'Ridge Regression (lags + calendar)': Ridge(alpha=1.0),
        'Random Forest (lags + calendar)': RandomForestRegressor(
            n_estimators=200, min_samples_leaf=2, random_state=random_state, n_jobs=-1),
        'Extra Trees (lags + calendar)': ExtraTreesRegressor(
            n_estimators=200, min_samples_leaf=2, random_state=random_state, n_jobs=-1),
    }

    results = {}
    preds = {}
    fitted = {}
    alpha = 0.3
    for name, model in models.items():
        if model is None and name != 'Naive (last value)':
            continue
        if name == 'Naive (last value)':
            pred = Xte['lag_1'].to_numpy()
            fitted[name] = None
        elif model == 'moving_average':
            lag_cols = [f'lag_{k}' for k in range(1, min(3, lags) + 1)]
            pred = Xte[lag_cols].mean(axis=1).to_numpy()
            fitted[name] = None
        elif model == 'exp_smoothing':
            level = float(ytr.iloc[0])
            for val in ytr.iloc[1:]:
                level = alpha * float(val) + (1 - alpha) * level
            sequence = []
            for val in yte:
                sequence.append(level)
                level = alpha * float(val) + (1 - alpha) * level
            pred = np.asarray(sequence)
            fitted[name] = {'alpha': alpha, 'level': level}
        elif model == 'seasonal_naive':
            pred = Xte['seasonal_lag'].to_numpy()
            fitted[name] = {'seasonal_period': seasonal_period, 'horizon': horizon}
        else:
            pipe = Pipeline([('model', model)])
            pipe.fit(Xtr, ytr)
            pred = pipe.predict(Xte)
            fitted[name] = pipe

        preds[name] = np.asarray(pred)
        actual = np.asarray(yte, dtype=float)
        nonzero = np.abs(actual) > 1e-12
        mape = (float(np.mean(np.abs((actual[nonzero] - preds[name][nonzero]) / actual[nonzero])) * 100)
                if nonzero.any() else float('nan'))
        results[name] = {
            'MAE': float(mean_absolute_error(yte, pred)),
            'RMSE': float(np.sqrt(mean_squared_error(yte, pred))),
            'MAPE (%)': mape,
            'R²': float(r2_score(yte, pred)),
        }

    best = min(results, key=lambda name: results[name]['RMSE'])
    pred_frame = pd.DataFrame({
        'Date': test['target_date'].dt.strftime('%Y-%m-%d %H:%M'),
        'Actual': yte.to_numpy(),
        'Predicted': preds[best],
    })

    # The selected horizon means one forecast value at t + horizon * frequency.
    future_dates = [last_observed_date + delta * horizon]
    if best == 'Naive (last value)':
        future_values = [float(full_history[-1])]
    elif best == 'Moving Average (recent lags)':
        future_values = [float(np.mean(full_history[-min(3, lags):]))]
    elif best == 'Exponential Smoothing':
        level = float(full_history[0])
        for val in full_history[1:]:
            level = alpha * float(val) + (1 - alpha) * level
        future_values = [level]
    elif best == 'Seasonal Naive':
        seasonal_offset = seasonal_period - horizon
        future_values = [float(full_history[-(seasonal_offset + 1)])]
    else:
        model = fitted[best]
        future_date = future_dates[0]
        cal_row = _calendar_features([future_date])
        row = {f'lag_{k}': [full_history[-k]] for k in range(1, lags + 1)}
        row.update({column: [cal_row.iloc[0][column]] for column in cal_row.columns})
        if use_seasonal_lag:
            seasonal_offset = seasonal_period - horizon
            row['seasonal_lag'] = [full_history[-(seasonal_offset + 1)]]
        future_values = [float(model.predict(pd.DataFrame(row))[0])]

    future = pd.DataFrame({
        'Date': [date.strftime('%Y-%m-%d %H:%M') for date in future_dates],
        'Forecast': future_values,
    })

    return {
        'task': 'forecasting',
        'target': target,
        'date_column': date_column,
        'time_column': time_column,
        'features': feature_cols,
        'rows_used': len(work),
        'train_rows': len(train),
        'test_rows': len(test),
        'duplicate_timestamps_aggregated': duplicate_count,
        'frequency': str(delta),
        'seasonal_period': seasonal_period,
        'results': results,
        'best_model': best,
        'best_metrics': results[best],
        'predictions': pred_frame,
        'future_forecast': future,
        'forecast_horizon': horizon,
        'fitted_models': fitted,
        'best_pipeline': fitted.get(best) if isinstance(fitted.get(best), Pipeline) else None,
        'model_artifact': {
            'task': 'forecasting',
            'best_model': best,
            'fitted_model': fitted.get(best),
            'lags': lags,
            'horizon': horizon,
            'seasonal_period': seasonal_period,
            'date_column': date_column,
            'time_column': time_column,
            'frequency': str(delta),
        },
        'feature_importance': [],
        'confusion_matrix': None,
        'classification_report': None,
        'selection_metric': 'RMSE',
    }


def train_clustering(data: bytes, features=None, n_clusters: int=3, random_state: int=42):

    """Unsupervised clustering over explicitly selected numeric columns."""

    from sklearn.metrics import silhouette_score

    df=load_frame(data)

    selected=list(features) if features is not None else df.select_dtypes(include=np.number).columns.tolist()

    excluded=[]; usable=[]

    for col in selected:

        if col not in df.columns: continue

        if _is_identifier(col): excluded.append({'feature':col,'reason':'Likely identifier.'})

        elif not pd.api.types.is_numeric_dtype(df[col]): excluded.append({'feature':col,'reason':'Clustering currently requires numeric features.'})

        elif df[col].nunique(dropna=True)<=1: excluded.append({'feature':col,'reason':'Constant or all-missing feature.'})

        else: usable.append(col)

    if not usable: raise ValueError('Select at least one meaningful numeric feature for clustering.')

    X=df[usable].replace([np.inf,-np.inf],np.nan)

    prep=Pipeline([('imputer',SimpleImputer(strategy='median')),('scaler',StandardScaler())])

    Z=prep.fit_transform(X)

    if len(Z)<10: raise ValueError('Clustering needs at least 10 usable rows for a meaningful exploratory run.')

    if not 2 <= n_clusters < len(Z): raise ValueError('Choose a cluster count between 2 and number of rows minus 1.')

    models={'K-Means':KMeans(n_clusters=n_clusters,n_init=10,random_state=random_state),

            'Agglomerative':AgglomerativeClustering(n_clusters=n_clusters),

            'DBSCAN':DBSCAN(eps=0.8,min_samples=5)}

    results={}; labels_map={};

    for name,model in models.items():

        labels=model.fit_predict(Z); labels_map[name]=labels

        valid=labels!=-1; unique=np.unique(labels[valid])

        sil=float(silhouette_score(Z[valid],labels[valid])) if len(unique)>=2 and valid.sum()>len(unique) else None

        results[name]={'Clusters':int(len(unique)),'Noise points':int((labels==-1).sum()),'Silhouette':sil}

    valid_scores=[(name,v['Silhouette']) for name,v in results.items() if v['Silhouette'] is not None]

    best=max(valid_scores,key=lambda x:x[1])[0] if valid_scores else 'K-Means'

    out=df[usable].copy(); out['Cluster']=labels_map[best]

    projection = None

    if Z.shape[1] >= 2:

        from sklearn.decomposition import PCA

        coords=PCA(n_components=2,random_state=random_state).fit_transform(Z)

        projection=pd.DataFrame({'Component 1':coords[:,0],'Component 2':coords[:,1],

                                 'Cluster':labels_map[best].astype(str)})

    return {'task':'clustering','features':usable,'excluded_features':excluded,'rows_used':len(df),

            'results':results,'best_model':best,'best_metrics':results[best],

            'cluster_assignments':out,'cluster_projection':projection,'labels':labels_map,'fitted_models':models,

            'preprocessor':prep,'selection_metric':'Silhouette (exploratory)'}
