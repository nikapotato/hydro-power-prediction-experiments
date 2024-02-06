# %%
import pandas as pd
import tsfresh
from urllib.request import urlopen
from io import BytesIO
from zipfile import ZipFile

#%%

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00501/PRSA2017_Data_20130301-20170228.zip"
r = urlopen(url)
zf = ZipFile(BytesIO(r.read()))

df = pd.DataFrame()
for file in zf.infolist():
    if file.filename.endswith('.csv'):
        df = df.append(pd.read_csv(zf.open(file)))

df['timestamp'] = pd.to_datetime(df[["year", "month", "day", "hour"]])
df.drop(columns=['No'], inplace=True)
df.sort_values(by=['timestamp', 'station']).head(10)

# %%

df.isnull().sum()
ts_df = df.dropna()

# %%
# columns
floats = ['PM2.5',
'PM10',
'SO2',
'NO2',
'CO',
'O3',
'TEMP',
'PRES',
'DEWP',
'RAIN',
'WSPM']

ts_df.loc[:,floats] = ts_df[floats].astype(float)

# %%
df_features = tsfresh.extract_features(ts_df[['timestamp'] + ['station'] + floats], column_id='station', column_sort='timestamp',
                                       default_fc_parameters=tsfresh.feature_extraction.MinimalFCParameters())
df_features.columns

