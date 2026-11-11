import hashlib
import pickle
import warnings
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import (confusion_matrix, classification_report, accuracy_score,
    balanced_accuracy_score, precision_recall_fscore_support, f1_score,
    roc_curve, precision_recall_curve, auc)
from analysis import MAX_BYTES, MAX_ROWS, MAX_COLUMNS, DataError, read_csv
from enhancements import enrich_profile
from reporting import report_html
from ml_engine import load_frame, train_model, train_forecast, train_clustering, _is_identifier

PALETTE = ['#35C6B4', '#6EA8FE', '#B69CFF', '#FFB86B', '#FF7B9C', '#8BD49C', '#F2D16B']


def date_ratio(series):
    if pd.api.types.is_numeric_dtype(series):
        return 0.0
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        return pd.to_datetime(series, errors='coerce', utc=True).notna().mean()


def style_fig(fig, height=None):
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        colorway=PALETTE, font=dict(family='Inter, Segoe UI, sans-serif', color='#E8EEF8'),
        title_font=dict(size=17, color='#F4F7FC'),
        legend_title_font=dict(color='#D8E2F0'),
        margin=dict(l=25, r=25, t=65, b=35),
        height=height or 430,
    )
    fig.update_xaxes(gridcolor='rgba(148,163,184,.16)', zerolinecolor='rgba(148,163,184,.2)')
    fig.update_yaxes(gridcolor='rgba(148,163,184,.16)', zerolinecolor='rgba(148,163,184,.2)')
    return fig


def default_target(df, task, numeric_cols):
    candidates = [c for c in df.columns if not _is_identifier(c)]
    preferred_names = ['cnt', 'median_house_value', 'points', 'glyhb', 'math score',
                       'revenue', 'close', 'adj close', 'price', 'survived', 'y']
    if task == 'Classification':
        binary = [c for c in candidates if df[c].nunique(dropna=True) == 2]
        return next((c for c in preferred_names if c in binary), binary[0] if binary else candidates[0])
    numeric = [c for c in numeric_cols if c in candidates and df[c].nunique(dropna=True) > 1]
    return next((c for c in preferred_names if c in numeric), numeric[0] if numeric else (candidates[0] if candidates else df.columns[0]))


DEMO = b'''date,marketing_channel,orders,revenue,ad_spend,customer_type\n2026-01-01,Web,31,1550,240,New\n2026-01-02,Referral,24,1320,180,Returning\n2026-01-03,Email,18,920,110,Returning\n2026-01-04,Web,36,1820,260,New\n2026-01-05,Social,22,1180,160,New\n2026-01-06,Referral,29,1490,205,Returning\n2026-01-07,Email,20,1010,130,New\n2026-01-08,Web,41,2100,300,Returning\n2026-01-09,Social,25,1290,170,New\n2026-01-10,Referral,33,1710,230,Returning\n2026-01-11,Web,38,1980,280,New\n2026-01-12,Email,21,1080,145,Returning\n2026-01-13,Social,27,1410,190,New\n2026-01-14,Referral,35,1790,250,Returning\n2026-01-15,Web,44,2260,320,New\n2026-01-16,Email,23,1170,155,Returning\n2026-01-17,Social,30,1580,215,New\n2026-01-18,Referral,37,1910,270,Returning\n2026-01-19,Web,46,2390,340,New\n2026-01-20,Email,26,1340,180,Returning\n2026-01-21,Social,32,1660,225,New\n2026-01-22,Referral,40,2070,290,Returning\n2026-01-23,Web,49,2520,360,New\n2026-01-24,Email,28,1450,195,Returning\n2026-01-25,Social,34,1770,240,New\n2026-01-26,Referral,42,2180,305,Returning\n2026-01-27,Web,52,2680,380,New\n2026-01-28,Email,30,1530,205,Returning\n2026-01-29,Social,36,1880,255,New\n2026-01-30,Referral,45,2310,325,Returning\n2026-01-31,Web,55,2840,400,New\n2026-02-01,Email,32,1640,220,Returning\n2026-02-02,Social,39,2020,275,New\n2026-02-03,Referral,47,2440,340,Returning\n2026-02-04,Web,58,2990,420,New\n2026-02-05,Email,34,1750,235,Returning\n2026-02-06,Social,41,2130,290,New\n2026-02-07,Referral,50,2580,360,Returning\n2026-02-08,Web,61,3150,440,New\n2026-02-09,Email,36,1850,250,Returning\n2026-02-10,Social,43,2250,305,New\n2026-02-11,Referral,53,2730,380,New\n'''

st.set_page_config(page_title='Data Story | ML Workbench', page_icon='📊', layout='wide')
st.markdown('''<style>
:root{--ds-teal:#35C6B4;--ds-blue:#6EA8FE;--ds-violet:#B69CFF;--ds-panel:#192334;--ds-border:rgba(148,163,184,.20)}
.stApp{background:radial-gradient(circle at 8% 0%,rgba(53,198,180,.12),transparent 28%),radial-gradient(circle at 95% 5%,rgba(182,156,255,.13),transparent 26%),#0D1421;color:#E8EEF8}
.block-container{max-width:1480px;padding-top:1.7rem;padding-bottom:4rem}
h1,h2,h3{letter-spacing:-.025em}
h1{background:linear-gradient(90deg,#F4F7FC 0%,#7DE0D2 48%,#B69CFF 100%);-webkit-background-clip:text;-webkit-text-fill-color:transparent}
h2{color:#EAF1FB} h3{color:#AEEBE2}
[data-testid="stMetric"]{background:linear-gradient(145deg,rgba(53,198,180,.12),rgba(110,168,254,.08));border:1px solid var(--ds-border);padding:16px 18px;border-radius:16px}
[data-testid="stMetricLabel"]{color:#AFC0D4}[data-testid="stMetricValue"]{color:#F4F7FC}
section[data-testid="stSidebar"]{background:linear-gradient(180deg,#111D2D,#151A2B);border-right:1px solid var(--ds-border)}
.stButton>button,.stDownloadButton>button{border:0;border-radius:11px;background:linear-gradient(100deg,#159D91,#477FE5);color:white;font-weight:650;box-shadow:0 7px 22px rgba(36,125,170,.20);transition:transform .15s ease,filter .15s ease}
.stButton>button:hover,.stDownloadButton>button:hover{filter:brightness(1.12);transform:translateY(-1px);color:white}
[data-baseweb="select"]>div,[data-testid="stTextInput"] input,[data-testid="stNumberInput"] input{background:#1B2638;border-color:var(--ds-border);border-radius:10px}
[data-baseweb="tag"] {
    background-color: #166534 !important;
    border: 1px solid #15803D !important;
    color: #FFFFFF !important;
}

[data-baseweb="tag"] * {
    color: #FFFFFF !important;
    fill: #FFFFFF !important;
}
[data-testid="stExpander"]{border:1px solid var(--ds-border);border-radius:13px;background:rgba(25,35,52,.45)}
[data-testid="stDataFrame"]{border:1px solid var(--ds-border);border-radius:12px;overflow:hidden}
hr{border-color:var(--ds-border)}
[data-testid="stAlert"]{border-radius:12px}
</style>''', unsafe_allow_html=True)

st.title('Data Story')
st.markdown('**From client CSV to data insights, trained models and actionable prediction outputs.**')
st.caption('A local-first analytics and ML workbench for freelancers and businesses.')
with st.sidebar:
    st.header('Workspace')
    st.write('🛡️ Local processing · uploaded data is not sent to an external model provider.')
    st.caption('CSV limits: 5 MB · 10,000 rows · 30 columns')
    if st.button('Load built-in demo dataset', use_container_width=True):
        st.session_state['demo_data'] = DEMO
    if st.button('Clear saved results', use_container_width=True):
        st.session_state.pop('training_result', None)
    st.caption('Prototype only: no user accounts, access controls, multi-user isolation or production security review. Use synthetic or non-sensitive data.')

upload = st.file_uploader('Upload your dataset (CSV)', type=['csv'])
data = upload.getvalue() if upload else st.session_state.get('demo_data')
if data is None:
    st.info('Upload a CSV or select **Load built-in demo dataset** from the sidebar to begin.')
    st.stop()
if len(data) > MAX_BYTES:
    st.error('File exceeds the 5 MB limit.'); st.stop()
signature = hashlib.sha256(data).hexdigest()
if st.session_state.get('result_signature') != signature:
    st.session_state.pop('training_result', None)
    st.session_state['result_signature'] = signature
try:
    profile = enrich_profile(data)
    df = load_frame(data)
except Exception as exc:
    st.error(f'Could not read this CSV: {exc}'); st.stop()
if len(df) > MAX_ROWS or len(df.columns) > MAX_COLUMNS:
    st.error(f'This prototype supports up to {MAX_ROWS:,} rows and {MAX_COLUMNS} columns.'); st.stop()

st.success(f'✓ Dataset loaded: {len(df):,} rows × {len(df.columns)} columns')
cols = st.columns(4)
cols[0].metric('Rows', f'{len(df):,}')
cols[1].metric('Columns', len(df.columns))
cols[2].metric('Missing cells', f'{int(df.isna().sum().sum()):,}')
cols[3].metric('Duplicate rows', f'{int(df.duplicated().sum()):,}')

with st.expander('Dataset preview and column types', expanded=True):
    st.dataframe(df.head(12), use_container_width=True, hide_index=True)
    type_table = pd.DataFrame({'Column':df.columns,
        'Detected type':[('Numeric' if pd.api.types.is_numeric_dtype(df[c]) else 'Date/time' if date_ratio(df[c])>.85 else 'Categorical / text') for c in df.columns],
        'Unique values':[df[c].nunique(dropna=True) for c in df.columns],
        'Missing':[int(df[c].isna().sum()) for c in df.columns]})
    st.dataframe(type_table, use_container_width=True, hide_index=True)

st.header('1. Explore the data')
left, right = st.columns(2)
numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
cat_cols = [c for c in df.columns if c not in numeric_cols and df[c].nunique(dropna=True)<=30]
with left:
    if numeric_cols:
        metric = st.selectbox('Numeric column to inspect', numeric_cols, key='eda_metric')
        fig = px.histogram(df, x=metric, nbins=30, title=f'Distribution of {metric}', marginal='box', color_discrete_sequence=[PALETTE[0]])
        st.plotly_chart(style_fig(fig, 390), use_container_width=True)
    else: st.info('No numeric columns detected.')
with right:
    if cat_cols:
        category = st.selectbox('Category to compare', cat_cols, key='eda_category')
        counts = df[category].astype('string').fillna('(Missing)').value_counts().head(12).rename_axis(category).reset_index(name='Rows')
        chart_kind = st.radio('Chart type', ['Bar chart','Pie chart'], horizontal=True, key='eda_chart')
        fig = (px.bar(counts, x=category, y='Rows', title=f'Rows by {category}', color=category,
                      color_discrete_sequence=PALETTE) if chart_kind=='Bar chart' else
               px.pie(counts, names=category, values='Rows', title=f'Share by {category}', hole=.42, color_discrete_sequence=PALETTE))
        st.plotly_chart(style_fig(fig, 390), use_container_width=True)
    else: st.info('No low-cardinality categorical columns were detected.')
if len(numeric_cols)>=2:
    st.subheader('Relationships between numeric variables')
    xcol, ycol = st.columns(2)
    xvar = xcol.selectbox('X axis', numeric_cols, index=0, key='scatter_x')
    yvar = ycol.selectbox('Y axis', numeric_cols, index=1, key='scatter_y')
    fig = px.scatter(df, x=xvar, y=yvar, opacity=.55, title=f'{yvar} vs {xvar}', color_discrete_sequence=[PALETTE[1]])
    fig.update_traces(marker=dict(size=6))
    st.plotly_chart(style_fig(fig, 420), use_container_width=True)
if profile.get('time_charts'):
    st.subheader('Time-based trends')
    for chart in profile['time_charts']:
        points = chart['points']
        tdf = pd.DataFrame({'Date':[p['label'] for p in points], 'Value':[p['value'] for p in points]})
        fig = px.line(tdf, x='Date', y='Value', title=chart['title'], color_discrete_sequence=[PALETTE[0]])
        fig.update_traces(line=dict(width=2.4), mode='lines')
        st.plotly_chart(style_fig(fig, 380), use_container_width=True)
with st.expander('Data quality checks'):
    for warning in profile['warnings']:
        st.write('• ' + warning)
    if not profile['warnings']:
        st.write('No basic missing-value or duplicate-row warnings were detected.')
    st.caption('These checks are screening signals, not a fairness, privacy or deployment certification.')

st.header('2. Build and evaluate a machine-learning model')
st.write('Select a task and target. Data Story compares suitable baselines and models, evaluates held-out data, and provides prediction outputs.')
task = st.selectbox('What type of problem?', ['Auto detect','Classification','Regression','Forecasting','Clustering'],
    help='Classification predicts a class; regression predicts a number; forecasting predicts future time steps; clustering explores groups without a target.')
candidate_targets = list(df.columns)
non_id_numeric = [c for c in numeric_cols if not _is_identifier(c) and df[c].nunique(dropna=True)>1]
result = None

if task == 'Clustering':
    default_cluster_features = non_id_numeric[:min(8, len(non_id_numeric))]
    cluster_features = st.multiselect('Numeric input features for clustering', non_id_numeric, default=default_cluster_features,
        help='Clustering is unsupervised. Select meaningful numeric measurements; identifiers and text columns are excluded.')
    n_clusters = st.slider('Number of clusters for K-Means and Agglomerative', 2, 10, 3)
    train_button = st.button('Discover clusters', type='primary')
    if train_button:
        try:
            with st.spinner('Scaling numeric features and comparing clustering methods...'):
                result = train_clustering(data, features=cluster_features, n_clusters=n_clusters)
            st.session_state['training_result'] = result
        except Exception as exc: st.error(str(exc))
else:
    target_default = default_target(df, task if task in ('Classification','Regression') else 'Regression', numeric_cols)
    target = st.selectbox('Target (the value the model should predict)', candidate_targets,
        index=candidate_targets.index(target_default) if target_default in candidate_targets else 0,
        help='Do not select an ID or a field that is only known after the prediction is made.')
    if task == 'Forecasting':
        date_candidates = [c for c in df.columns if c != target and date_ratio(df[c]) >= .8]
        if not date_candidates:
            st.warning('No date/time column could be identified. Forecasting requires a genuine time column.')
        else:
            date_default = next((c for c in date_candidates if c.lower() in ('datetime','timestamp','dteday','date')), date_candidates[0])
            date_col = st.selectbox('Date column', date_candidates, index=date_candidates.index(date_default))
            time_candidates = [c for c in df.columns if c not in (target,date_col) and
                               (c.lower() in ('hr','hour','time','minute') or 'hour' in c.lower() or c.lower().endswith('_hr'))]
            time_options = ['None'] + time_candidates
            time_col_choice = st.selectbox('Optional time-of-day column', time_options,
                index=(time_options.index('hr') if 'hr' in time_options else 0),
                help='Combine a date with an hour/minute column for hourly datasets such as Bike Sharing.')
            time_col = None if time_col_choice == 'None' else time_col_choice
            default_lags = 24 if time_col else 5
            lag_count = st.slider('Past time steps used as input (lags)', 2, 168, default_lags)
            horizon = st.number_input('Forecast horizon (number of observed steps ahead)', min_value=1, max_value=24, value=1)
            seasonal_default = 24 if time_col else 7
            seasonal_period = st.number_input('Seasonal period (0 disables seasonal-naive model)', min_value=0, max_value=168, value=seasonal_default,
                help='For hourly data, 24 represents one day. For daily data, 7 represents one week. Confirm the frequency before relying on this baseline.')
            train_button = st.button('Train forecasting models', type='primary')
            if train_button:
                try:
                    with st.spinner('Building chronological features and evaluating forecasting baselines/models...'):
                        result = train_forecast(data, target, date_col, lags=lag_count, horizon=int(horizon),
                                                time_column=time_col, seasonal_period=int(seasonal_period))
                    st.session_state['training_result'] = result
                except Exception as exc: st.error(str(exc))
    else:
        resolved_task = {'Auto detect':'Auto','Classification':'classification','Regression':'regression'}[task]
        if any(date_ratio(df[c]) >= .8 for c in df.columns):
            st.warning('A date/time column is present. For a future-value objective, use Forecasting and a chronological split instead of random Regression.')
        default_features = [c for c in df.columns if c != target and not _is_identifier(c)
                            and df[c].nunique(dropna=True)>1 and date_ratio(df[c])<.8]
        features = st.multiselect('Input features (columns used to make the prediction)',
            [c for c in df.columns if c != target], default=default_features,
            help='Identifier, date and high-cardinality text columns may be excluded automatically. Avoid target leakage and post-outcome fields.')
        test_size = st.slider('Test data proportion', 0.1, 0.4, 0.2, 0.05)
        run_cv = st.checkbox('Run optional 3-fold cross-validation (slower)', value=False)
        train_button = st.button('Train and compare models', type='primary')
        if train_button:
            try:
                with st.spinner('Preprocessing, training model candidates and evaluating held-out rows...'):
                    result = train_model(data, target, task=resolved_task, features=features,
                                         test_size=test_size, run_cv=run_cv)
                st.session_state['training_result'] = result
            except Exception as exc: st.error(str(exc))

result = st.session_state.get('training_result')
if result:
    st.divider(); st.header('3. Model performance and outputs')
    if result['task'] == 'clustering':
        st.success(f"Highest valid silhouette among the compared methods: **{result['best_model']}**")
        st.caption('Silhouette is an internal separation score, not prediction accuracy or proof that real business segments exist.')
        a,b = st.columns(2); a.metric('Rows clustered', f"{result['rows_used']:,}"); b.metric('Features used', len(result['features']))
        st.subheader('Clustering comparison')
        st.dataframe(pd.DataFrame(result['results']).T.reset_index(names='Method'), use_container_width=True, hide_index=True)
        if result.get('cluster_projection') is not None:
            fig = px.scatter(result['cluster_projection'], x='Component 1', y='Component 2', color='Cluster',
                             title='Cluster map (PCA projection)', opacity=.72, color_discrete_sequence=PALETTE)
            fig.update_traces(marker=dict(size=6))
            st.plotly_chart(style_fig(fig, 480), use_container_width=True)
        st.subheader('Cluster assignments')
        st.dataframe(result['cluster_assignments'].head(100), use_container_width=True, hide_index=True)
        st.download_button('Download cluster assignments (CSV)', result['cluster_assignments'].to_csv(index=False).encode(),
                           file_name='data_story_cluster_assignments.csv', mime='text/csv')
        cluster_artifact = pickle.dumps({'task':'clustering','model':result['fitted_models'][result['best_model']],
                                         'preprocessor':result['preprocessor'],'features':result['features']})
        st.download_button('Download clustering model (.pkl)', cluster_artifact, file_name='data_story_cluster_model.pkl', mime='application/octet-stream')
    else:
        st.success(f"Best model on this held-out test set: **{result['best_model']}** · selected by {result.get('selection_metric','the primary metric')}")
        a,b,c = st.columns(3); a.metric('Rows used', f"{result['rows_used']:,}"); b.metric('Training rows', f"{result['train_rows']:,}"); c.metric('Test rows', f"{result['test_rows']:,}")
        metrics = pd.DataFrame(result['results']).T.reset_index(names='Model')
        st.subheader('Model comparison')
        st.caption('Input columns actually used: ' + ', '.join(result['features']))
        for warning in result.get('target_warnings', []):
            st.warning(warning)
        st.dataframe(metrics, use_container_width=True, hide_index=True)
        if result.get('excluded_features'):
            with st.expander('Columns not used and why'):
                st.dataframe(pd.DataFrame(result['excluded_features']), use_container_width=True, hide_index=True)
        if result.get('cross_validation'):
            st.subheader('Cross-validation stability')
            cv_rows = []
            for name, values in result['cross_validation'].items():
                mean_score, std_score = values.get('mean'), values.get('std')
                if values.get('error'):
                    note = f"CV failed: {values['error']}"
                elif mean_score is None or std_score is None:
                    note = 'Not available'
                elif abs(mean_score) < 1e-12:
                    note = 'Variation cannot be expressed relative to a near-zero mean'
                else:
                    relative_variation = abs(std_score / mean_score)
                    if relative_variation <= 0.01:
                        note = 'Low fold-to-fold variation'
                    elif relative_variation <= 0.05:
                        note = 'Moderate fold-to-fold variation'
                    else:
                        note = 'High fold-to-fold variation; inspect individual folds'
                cv_rows.append({'Model': name, 'Mean score': mean_score,
                                'Std. deviation': std_score, 'Note': note})
            st.caption(result.get('cross_validation_method',''))
            st.dataframe(pd.DataFrame(cv_rows), use_container_width=True, hide_index=True)
        metric_names = [c for c in metrics.columns if c != 'Model']
        preferred = result.get('selection_metric', 'F1 (weighted)' if result['task']=='classification' else 'RMSE')
        if preferred in metrics.columns:
            fig = px.bar(metrics, x='Model', y=preferred, title=f'Model comparison — {preferred}', text_auto='.4f', color='Model', color_discrete_sequence=PALETTE)
            st.plotly_chart(style_fig(fig, 410), use_container_width=True)
        preds = result['predictions'].copy()
        if result['task'] == 'classification':
            probabilities = result.get('probabilities')
            classes = list(result.get('classes', []))
            if not classes:
                classes = sorted(pd.Series(result['y_test']).unique().tolist())

            # Standard evaluation always uses the selected model's original
            # predict() output, exactly as used to calculate the comparison table.
            y_true = np.asarray(result['y_test'])
            standard_pred = preds['Predicted'].to_numpy()
            labels = classes
            cm = confusion_matrix(y_true, standard_pred, labels=labels)

            st.subheader('Standard held-out evaluation')
            st.caption(
                f"Selected model: {result['best_model']} · "
                f"Selection metric: {result.get('selection_metric', 'primary metric')} · "
                f"Evaluation: untouched {result['test_rows']:,}-row test split · "
                'Predictions: model.predict()'
            )
            standard_acc = accuracy_score(y_true, standard_pred)
            standard_pr, standard_rc, _, _ = precision_recall_fscore_support(
                y_true, standard_pred, average='weighted', zero_division=0
            )
            standard_f1 = f1_score(y_true, standard_pred, average='weighted', zero_division=0)
            sm1, sm2, sm3, sm4 = st.columns(4)
            sm1.metric('Accuracy', f'{standard_acc:.4f}')
            sm2.metric('Precision (weighted)', f'{standard_pr:.4f}')
            sm3.metric('Recall (weighted)', f'{standard_rc:.4f}')
            sm4.metric('F1 (weighted)', f'{standard_f1:.4f}')

            cm_fig = px.imshow(
                cm, x=labels, y=labels, text_auto=True,
                color_continuous_scale=['#162235', '#277F91', '#9CE9DF'],
                labels=dict(x='Predicted class', y='Actual class', color='Count'),
                title=f"{result['best_model']}: actual vs predicted classes"
            )
            st.plotly_chart(style_fig(cm_fig, 430), use_container_width=True)
            report = pd.DataFrame(classification_report(
                y_true, standard_pred, labels=labels, output_dict=True,
                zero_division=0
            )).T
            st.subheader('Per-class performance — standard predictions')
            st.dataframe(report, use_container_width=True)

            # Threshold analysis is deliberately separate: it must not replace
            # standard predictions or change the model-comparison metrics.
            if probabilities is not None and len(classes) == 2:
                with st.expander('Optional decision-threshold analysis', expanded=False):
                    st.caption(
                        'These results are a separate what-if analysis. Changing the threshold '
                        'does not change the standard evaluation or model-comparison table.'
                    )
                    positive = st.selectbox(
                        'Positive class for threshold analysis', classes,
                        index=1, key='threshold_positive_class'
                    )
                    threshold = st.slider(
                        'Decision threshold', 0.05, 0.95, 0.50, 0.01,
                        help='Predict the selected positive class when its estimated probability '
                             'is at least this threshold.', key='decision_threshold'
                    )
                    positive_idx = classes.index(positive)
                    p_pos = probabilities[:, positive_idx]
                    negative = classes[1 - positive_idx]
                    threshold_pred = np.where(p_pos >= threshold, positive, negative)
                    threshold_cm = confusion_matrix(y_true, threshold_pred, labels=labels)
                    tpr, trec, _, _ = precision_recall_fscore_support(
                        y_true, threshold_pred, average='weighted', zero_division=0
                    )
                    threshold_macro = f1_score(
                        y_true, threshold_pred, average='macro', zero_division=0
                    )
                    fn = int(threshold_cm[positive_idx, :].sum() - threshold_cm[positive_idx, positive_idx])
                    fp = int(threshold_cm[:, positive_idx].sum() - threshold_cm[positive_idx, positive_idx])
                    tm1, tm2, tm3, tm4, tm5 = st.columns(5)
                    tm1.metric('Threshold accuracy', f'{accuracy_score(y_true, threshold_pred):.4f}')
                    tm2.metric('Weighted precision', f'{tpr:.4f}')
                    tm3.metric('Weighted recall', f'{trec:.4f}')
                    tm4.metric('Macro F1', f'{threshold_macro:.4f}')
                    tm5.metric('False negatives / positives', f'{fn} / {fp}')
                    threshold_fig = px.imshow(
                        threshold_cm, x=labels, y=labels, text_auto=True,
                        color_continuous_scale=['#162235', '#277F91', '#9CE9DF'],
                        labels=dict(x='Predicted class', y='Actual class', color='Count'),
                        title=f'Threshold-adjusted confusion matrix (threshold={threshold:.2f})'
                    )
                    st.plotly_chart(style_fig(threshold_fig, 390), use_container_width=True)
                    threshold_report = pd.DataFrame(classification_report(
                        y_true, threshold_pred, labels=labels, output_dict=True,
                        zero_division=0
                    )).T
                    st.dataframe(threshold_report, use_container_width=True)

                    # Curves use the chosen positive class and its probability scores.
                    fpr, tpr_curve, _ = roc_curve((y_true == positive).astype(int), p_pos)
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=fpr, y=tpr_curve, mode='lines', name='ROC curve',
                        line=dict(color=PALETTE[0], width=3)
                    ))
                    fig.add_trace(go.Scatter(
                        x=[0, 1], y=[0, 1], mode='lines', name='Random baseline',
                        line=dict(color='#94A3B8', dash='dash')
                    ))
                    fig.update_layout(
                        title='ROC curve', xaxis_title='False positive rate',
                        yaxis_title='True positive rate'
                    )
                    st.plotly_chart(style_fig(fig, 390), use_container_width=True)
                    precision_curve, recall_curve, _ = precision_recall_curve(
                        (y_true == positive).astype(int), p_pos
                    )
                    prfig = go.Figure(go.Scatter(
                        x=recall_curve, y=precision_curve, mode='lines',
                        name='Precision–Recall', line=dict(color=PALETTE[2], width=3)
                    ))
                    prfig.update_layout(
                        title='Precision–Recall curve', xaxis_title='Recall',
                        yaxis_title='Precision'
                    )
                    st.plotly_chart(style_fig(prfig, 390), use_container_width=True)
                    st.caption(f'ROC and PR curves treat {positive!r} as the positive class.')

            st.caption(
                'The standard confusion matrix and classification report use the same '
                'held-out predictions as the model-comparison metrics. Threshold analysis, '
                'when expanded, is a separate what-if evaluation.'
            )
        elif result['task'] == 'forecasting':
            st.info(f"Detected/constructed frequency: {result.get('frequency')}. Duplicate timestamps aggregated: {result.get('duplicate_timestamps_aggregated',0)}. Calendar features and lag features are used; future external weather or business variables are not assumed known.")
            trend = preds.melt(id_vars='Date', value_vars=['Actual','Predicted'], var_name='Series', value_name='Value')
            fig = px.line(trend, x='Date', y='Value', color='Series', title='Chronological holdout: actual vs predicted',
                          color_discrete_map={'Actual':PALETTE[1],'Predicted':PALETTE[0]})
            fig.update_traces(mode='lines', line=dict(width=2.0))
            st.plotly_chart(style_fig(fig, 500), use_container_width=True)
            preds['Residual'] = preds['Actual']-preds['Predicted']
            fig = px.histogram(preds, x='Residual', nbins=35, title='Forecast error (residual) distribution', color_discrete_sequence=[PALETTE[2]])
            st.plotly_chart(style_fig(fig, 360), use_container_width=True)
        else:
            fig = px.scatter(preds, x='Actual', y='Predicted', title='Held-out actual values vs predictions', opacity=.48,
                             color_discrete_sequence=[PALETTE[1]])
            fig.update_traces(marker=dict(size=6))
            lo = float(min(preds['Actual'].min(), preds['Predicted'].min())); hi = float(max(preds['Actual'].max(), preds['Predicted'].max()))
            fig.add_shape(type='line', x0=lo, y0=lo, x1=hi, y1=hi, line=dict(color=PALETTE[0], width=2, dash='dash'))
            fig.add_annotation(x=hi, y=hi, text='Perfect prediction', showarrow=False, xanchor='right', yanchor='bottom', font=dict(color=PALETTE[0]))
            st.plotly_chart(style_fig(fig, 470), use_container_width=True)
            preds['Residual'] = preds['Actual']-preds['Predicted']
            c1,c2 = st.columns(2)
            hist = px.histogram(preds, x='Residual', nbins=35, title='Residual distribution', color_discrete_sequence=[PALETTE[2]])
            c1.plotly_chart(style_fig(hist, 350), use_container_width=True)
            resid = px.scatter(preds, x='Predicted', y='Residual', opacity=.48, title='Residuals vs predicted', color_discrete_sequence=[PALETTE[3]])
            resid.update_traces(marker=dict(size=6)); resid.add_hline(y=0, line_dash='dash', line_color=PALETTE[0])
            c2.plotly_chart(style_fig(resid, 350), use_container_width=True)
        if result.get('feature_importance'):
            st.subheader('Most influential model features')
            imp = pd.DataFrame(result['feature_importance'])
            fig = px.bar(imp.sort_values('importance'), x='importance', y='feature', orientation='h', title='Model feature importance', color_discrete_sequence=[PALETTE[0]])
            st.plotly_chart(style_fig(fig, 430), use_container_width=True)
            st.caption('Importance is model-specific association, not causation.')
        st.subheader('Prediction output on held-out test rows')
        st.dataframe(preds.head(100), use_container_width=True, hide_index=True)
        st.download_button('Download all test predictions (CSV)', preds.to_csv(index=False).encode(),
            file_name='data_story_test_predictions.csv', mime='text/csv')
        if result['task']=='forecasting':
            st.subheader('Future forecast output')
            future = result['future_forecast']; st.dataframe(future, use_container_width=True, hide_index=True)
            fig = px.line(future, x='Date', y='Forecast', title='Forecasted future values', color_discrete_sequence=[PALETTE[0]])
            fig.update_traces(mode='lines', line=dict(width=2.5))
            st.plotly_chart(style_fig(fig, 360), use_container_width=True)
            st.download_button('Download forecast (CSV)', future.to_csv(index=False).encode(), file_name='data_story_forecast.csv', mime='text/csv')
            forecast_artifact = pickle.dumps(result.get('model_artifact', {'task':'forecasting','best_model':result['best_model']}))
            st.download_button('Download forecasting model/configuration (.pkl)', forecast_artifact, file_name='data_story_forecast_model.pkl', mime='application/octet-stream')
        else:
            st.subheader('Try a prediction with your own input')
            pipe = result['best_pipeline']; chosen = result['features']; model_df = load_frame(data)
            with st.form('custom_prediction'):
                st.caption('Initial values use the first available median/category in the uploaded data. Change them to test another case.')
                values = {}
                for col in chosen:
                    series = model_df[col]
                    if pd.api.types.is_numeric_dtype(series):
                        default = float(series.dropna().median()) if series.notna().any() else 0.0
                        values[col] = st.number_input(col, value=default, format='%.4f')
                    else:
                        options = series.dropna().astype(str).value_counts().index.tolist()[:100]
                        values[col] = st.selectbox(col, options or ['Unknown'])
                clicked = st.form_submit_button('Predict this example', type='primary')
            if clicked:
                try:
                    one = pd.DataFrame([values], columns=chosen)
                    prediction = pipe.predict(one)[0]
                    st.metric('Model prediction', str(prediction) if result['task']=='classification' else f'{float(prediction):,.4f}')
                except Exception as exc: st.error(f'Prediction failed: {exc}')
        try:
            artifact = pickle.dumps(result['best_pipeline']) if result.get('best_pipeline') is not None else None
            if artifact:
                st.download_button('Download trained model (.pkl)', artifact, file_name='data_story_model.pkl', mime='application/octet-stream')
        except Exception: st.caption('Model artifact export unavailable for this model; prediction and metric downloads remain available.')
        st.warning('Performance is specific to this dataset and evaluation split. It is not a guarantee of future accuracy. Check leakage, representativeness, time ordering and domain suitability before real use.')

st.header('4. Export the analysis report')
title = st.text_input('Report title', value='Client data and model report')
author = st.text_input('Prepared by', value='')
include_labels = st.checkbox('Include actual category labels in report', value=False)
report = report_html(profile, title, author, include_labels, training_result=st.session_state.get('training_result'))
st.download_button('Download analysis report (HTML)', report, file_name='data-story-report.html', mime='text/html')
st.caption('The standalone HTML report contains the descriptive profile and, when available, model comparison metrics and held-out prediction examples. Review it before sharing with a client.')
