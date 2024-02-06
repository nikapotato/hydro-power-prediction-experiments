import numpy as np
import pandas as pd
import matplotlib.pylab as plt

from tsfresh import extract_features, select_features
from tsfresh.utilities.dataframe_functions import roll_time_series, make_forecasting_frame
from tsfresh.utilities.dataframe_functions import impute

try:
    import pandas_datareader.data as web
except ImportError:
    print("You need to install the pandas_datareader. Run pip install pandas_datareader.")

from sklearn.linear_model import LinearRegression

# %%
df = web.DataReader("AAPL", 'stooq')["High"]
df.head()
plt.figure(figsize=(15, 6))
df.plot(ax=plt.gca())
plt.show()

# %%
df_melted = pd.DataFrame({"high": df.copy()})
df_melted["date"] = df_melted.index
df_melted["Symbols"] = "AAPL"

df_melted.head()

# %%
df_rolled = roll_time_series(df_melted, column_id="Symbols", column_sort="date",
                             max_timeshift=20, min_timeshift=5)

df_rolled.head()

'''
The resulting dataframe now consists of these "windows" stamped out of the original dataframe. 
For example all data with the id = (AAPL, 2020-07-14 00:00:00) comes from the original data of stock AAPL including the last 20 days until 2020-07-14:
'''

df_rolled[df_rolled["id"] == ("AAPL", pd.to_datetime("2020-07-14"))]

# is this reversed
df_melted[(df_melted["date"] <= pd.to_datetime("2020-07-14")) &
          (df_melted["date"] >= pd.to_datetime("2020-06-15")) &
          (df_melted["Symbols"] == "AAPL")]

# %%
'''
Extract features
'''

X = extract_features(df_rolled.drop("Symbols", axis=1),
                     column_id="id", column_sort="date", column_value="high",
                     impute_function=impute, show_warnings=False, n_jobs=6)

# %%
'''
remove the tuple index 
'''

X = X.set_index(X.index.map(lambda x: x[1]), drop=True)
X.index.name = "last_date"
X.head()

# %%
X
X.loc['2020-07-14']

# %% shift target
y = df_melted.set_index("date").sort_index().high.shift(-1)

# %% match the target, missing initial windows, missing last step due to shift
y = y[y.index.isin(X.index)]
X = X[X.index.isin(y.index)]

# %%
X[:"2018"]

X_train = X[:"2018"]
X_test = X["2019":]

y_train = y[:"2018"]
y_test = y["2019":]

X_train_selected = select_features(X_train, y_train)

# %%
ada = LinearRegression()

ada.fit(X_train_selected, y_train)

# %%
# test prediction

X_test_selected = X_test[X_train_selected.columns]

# NOTE
y_pred = pd.Series(ada.predict(X_test_selected), index=X_test_selected.index)

# %%
plt.figure(figsize=(15, 6))

y.plot(ax=plt.gca(), color='blue')
y_pred.plot(ax=plt.gca(), legend=None, marker=".", color='green')
plt.show()
