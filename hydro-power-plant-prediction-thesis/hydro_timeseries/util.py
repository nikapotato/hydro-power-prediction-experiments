import seaborn as sns
import matplotlib.pylab as plt
import pandas as pd
from sklearn.feature_selection import RFE, SelectKBest, f_regression
from sklearn.preprocessing import StandardScaler
from sktime.forecasting.model_evaluation import evaluate
from sktime.performance_metrics.forecasting import MeanSquaredError, MeanAbsolutePercentageError, MeanAbsoluteError
from statsmodels.tsa.stattools import adfuller
from hydro_timeseries.variables import Variables
import numpy as np
import statsmodels.api as sm

smape = MeanAbsolutePercentageError(symmetric=True)
rmse = MeanSquaredError(square_root=True)
mae = MeanAbsoluteError()

from sklearn.base import BaseEstimator, TransformerMixin

def get_sample_weights(X_train: pd.DataFrame, daily_discount_rate: float) -> np.array:
    dates = X_train.index.date
    day_deltas = (dates[-1] - dates).astype('timedelta64[D]').astype(float)
    weights = np.power(np.full(shape=(len(X_train),), fill_value=(1 - daily_discount_rate)), day_deltas)
    return weights

class IdentityTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, input_array, y=None):
        return self

    def transform(self, input_array, y=None):
        return input_array * 1

    def inverse_transform(self, input_array, y=None):
        return input_array * 1


def OLS_analysis(target_column: str, feature_columns:list , features: pd.DataFrame):
    df_ols = features[[target_column] + feature_columns].dropna()
    feature_sc = StandardScaler()
    target_sc = StandardScaler()

    scaled_target = target_sc.fit_transform(df_ols[[target_column]])
    scaled_features = feature_sc.fit_transform(df_ols[feature_columns])

    features_df = pd.DataFrame(scaled_features, index=df_ols.index, columns=df_ols[feature_columns].columns)
    target_df = pd.DataFrame(scaled_target, index=df_ols.index)

    mod = sm.OLS(target_df, sm.add_constant(features_df))
    fii = mod.fit()
    print(fii.summary2())

def cv_evaluate(forecaster, y, cv, X=None, strategy='refit', scoring=smape):
    df = evaluate(forecaster=forecaster, y=y, cv=cv, X=X, strategy=strategy, scoring=scoring, return_data=True)
    y_pred = pd.concat(df['y_pred'].values)
    y_test = pd.concat(df['y_test'].values)

    smape_test = smape(y_test, y_pred)
    rmse_test = rmse(y_test, y_pred)
    mae_test = mae(y_test, y_pred)
    return y_pred, y_test, smape_test, rmse_test, mae_test, df



# feature selection
def select_kbest_features(X_train, y_train, score_func=f_regression, k='all'):
    fs = SelectKBest(score_func=score_func, k=k)
    fs.fit(X_train, y_train)
    cols = fs.get_support(indices=True)
    selected_features = X_train.iloc[:, cols]
    return fs, selected_features

def rfe_var_eliminate(model, n_features_to_select, train, y):
    rfe = RFE(model, n_features_to_select=n_features_to_select, step=1)
    rfe = rfe.fit(train, y)


    # summarize the ranking of the attributes
    fea_rank_ = pd.DataFrame({'cols': train.columns, 'fea_rank': rfe.ranking_})
    fea_rank = fea_rank_.loc[fea_rank_.fea_rank > 0].sort_values(by=['fea_rank'], ascending=True)
    return fea_rank

def get_fea_imp_gbm(model, train_vars):
    fea_imp_ = pd.DataFrame({'cols': train_vars.columns, 'fea_imp': model.feature_importances_})
    fea_imp_ = fea_imp_.loc[fea_imp_.fea_imp > 0]
    fea_imp_.sort_values('fea_imp', ascending=False, inplace=True)
    fea_imp_.reset_index(drop=True, inplace=True)
    return fea_imp_

def generate_cyclical_features(df, col_name, period, start_num=0):
    '''
    http://blog.davidkaleko.com/feature-engineering-cyclical-features.html
    '''
    kwargs = {
        f'sin_{col_name}' : lambda x: np.sin(2*np.pi*(df[col_name]-start_num)/period),
        f'cos_{col_name}' : lambda x: np.cos(2*np.pi*(df[col_name]-start_num)/period)
             }
    return df.assign(**kwargs).drop(columns=[col_name])

def load_timeseries_csv(data_path: str, parse_dates='Date_Time', index_col='Date_Time') -> pd.DataFrame:
    data = pd.read_csv(data_path, parse_dates=[parse_dates], index_col=index_col)
    data.index = data.index.tz_localize(tz = None)
    return data

def add_mean_vars(df: pd.DataFrame) -> pd.DataFrame:
    prefixes = set([var[0] for var in df[Variables.meteo].columns.str.split('_')])
    for var in prefixes:
        df[var + '_mean'] = df.filter(like=var).mean(axis=1)

    # rename to make some sense
    df.rename(columns={'volumetric_mean':'volumetric_soil_water_mean', 'soil_mean': 'soil_moisture_mean'}, inplace=True)

    return df


def adfuller_test(ts, window=12):
    movingAverage = ts.rolling(window).mean()
    movingSTD = ts.rolling(window).std()

    plt.figure(figsize=(10, 6))
    orig = plt.plot(ts, color='cornflowerblue',
                    label='Original')
    mean = plt.plot(movingAverage, color='firebrick',
                    label='Rolling Mean')
    std = plt.plot(movingSTD, color='limegreen',
                   label='Rolling Std')
    plt.legend(loc='upper left')
    plt.title('Rolling Statistics', size=14)
    plt.show(block=False)

    adf = adfuller(ts, autolag='AIC')

    print('ADF Statistic: {}'.format(round(adf[0], 3)))
    print(f'p-value: {adf[1]}')
    print("##################################")
    print('Critical Values:')

    for key, ts in adf[4].items():
        print('{}: {}'.format(key, round(ts, 3)))
    print("##################################")

    if adf[0] > adf[4]["5%"]:
        print("ADF > Critical Values")
        print("Failed to reject null hypothesis, time series is non-stationary.")
    else:
        print("ADF < Critical Values")
        print("Reject null hypothesis, time series is stationary.")

def stationarity_check(TS, window=8):

    # Calculate rolling statistics
    rolmean = TS.rolling(window=window, center=False).mean()
    rolstd = TS.rolling(window=window, center=False).std()

    # Perform the Dickey Fuller Test
    dftest = adfuller(TS)

    # Print Dickey-Fuller test results
    print('Results of Dickey-Fuller Test:')

    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
    for key, value in dftest[4].items():
        dfoutput['Critical Value (%s)' % key] = value
    print(dfoutput)

    return None

def drop_col(df, cols):
    return df[df.columns.difference(cols)]



