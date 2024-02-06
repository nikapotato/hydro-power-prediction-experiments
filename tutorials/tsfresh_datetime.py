import pandas as pd
from tsfresh.feature_extraction import extract_features
# TimeBasedFCParameters contains all functions that use the Datetime index of the timeseries container
from tsfresh.feature_extraction.settings import TimeBasedFCParameters, MinimalFCParameters

df = pd.DataFrame({"id": ["a", "a", "a", "a", "b", "b", "b", "b"],
                   "value": [1, 2, 3, 1, 3, 1, 0, 8],
                   "kind": ["temperature", "temperature", "pressure", "pressure",
                            "temperature", "temperature", "pressure", "pressure"]},
                   index=pd.DatetimeIndex(
                       ['2019-03-01 10:04:00', '2019-03-01 10:50:00', '2019-03-02 00:00:00', '2019-03-02 09:04:59',
                        '2019-03-02 23:54:12', '2019-03-03 08:13:04', '2019-03-04 08:00:00', '2019-03-04 08:01:00']
                   ))
df = df.sort_index()
df

# %%
settings_time = TimeBasedFCParameters()
settings_time

# %%
X_tsfresh = extract_features(df, column_id="id", column_value='value', column_kind='kind',
                             default_fc_parameters=settings_time)
X_tsfresh.head()

# %%
settings_regular = {'linear_trend': [
  {'attr': 'pvalue'},
  {'attr': 'rvalue'},
  {'attr': 'intercept'},
  {'attr': 'slope'},
  {'attr': 'stderr'}
]}

X_tsfresh = extract_features(df, column_id="id", column_value='value', column_kind='kind',
                             default_fc_parameters=settings_regular)
X_tsfresh.head()

# %%
df.loc[:, 'timestamp'] = df.index

# %%

X_tsfresh = extract_features(df, column_id="id", column_value='value', column_kind='kind', default_fc_parameters=settings_regular)
X_tsfresh.columns

