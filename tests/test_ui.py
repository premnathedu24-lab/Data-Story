"""Static integration checks; browser-level Streamlit testing is a separate manual step."""
import ast
from pathlib import Path
from ml_engine import train_model, train_forecast
from reporting import report_html
from enhancements import enrich_profile
import pandas as pd
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
SOURCE=(ROOT/'app.py').read_text(encoding='utf-8')
assert 'Train and compare models' in SOURCE
assert 'Train forecasting models' in SOURCE
assert 'Download all test predictions (CSV)' in SOURCE
assert 'Predict this example' in SOURCE
assert 'Download trained model (.pkl)' in SOURCE
assert "'Dog, cat, horse'" not in SOURCE

# Small deterministic synthetic fixture validates report integration and both model paths.
n=80
frame=pd.DataFrame({'date':pd.date_range('2025-01-01',periods=n).astype(str),
                    'visits':np.arange(n)+10,'spend':np.arange(n)*2+5,
                    'channel':np.where(np.arange(n)%2,'Web','Email'),
                    'revenue':np.arange(n)*5+np.sin(np.arange(n))})
data=frame.to_csv(index=False).encode()
profile=enrich_profile(data)
reg=train_model(data,'revenue','regression',['visits','spend','channel'])
assert reg['predictions'].shape[0]==reg['test_rows']
profile['selected_target']='revenue'
html=report_html(profile,'Test report','Synthetic test',False,training_result=reg)
assert 'Machine-learning experiment results' in html
assert 'Held-out prediction examples' in html
forecast=train_forecast(data,'revenue','date',lags=4,horizon=1)
assert len(forecast['future_forecast'])==1
print('Static app checks and synthetic ML/report integration passed. Browser-level Streamlit testing remains a manual verification step.')
