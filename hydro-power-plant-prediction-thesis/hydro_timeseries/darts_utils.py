'''
Darts specific utils
'''

import pandas as pd
from numpy import mean
from pandas import DatetimeIndex

from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
from darts.metrics import rmse, mae, smape
from darts.models import AutoARIMA
from darts import timeseries
from matplotlib import pyplot as plt
from hydro_timeseries.plotting import plot_residuals, tsplot

def to_daily(ts: TimeSeries) -> TimeSeries:
    return ts.from_dataframe(ts.pd_dataframe().resample('D').mean())

def to_hourly(ts: TimeSeries) -> TimeSeries:
    return ts.from_dataframe(ts.pd_dataframe().resample('h').mean())

def to_timestep(ts: TimeSeries, timestep: str, aggregator = mean) -> TimeSeries:
    return ts.from_dataframe(ts.pd_dataframe().resample(timestep).agg(aggregator))

def ffill_to_minute(ts: TimeSeries, minute_index: DatetimeIndex) -> TimeSeries:
    df = ts.pd_dataframe()
    df_out = pd.DataFrame(columns=df.columns, index=minute_index)
    df_out.loc[df.index, :] = df.values
    ts_out = ts.from_dataframe(df_out.ffill())
    return ts_out

def plot_backtest(model, value_true: pd.Series, value_pred: pd.Series):
    true_ts = TimeSeries.from_series(value_true)
    pred_ts = TimeSeries.from_series(value_pred)


    fig = plt.figure(figsize=(12, 4))
    true_ts.plot()
    pred_ts.plot(label=f'{model.__class__.__name__} SMAPE={smape(true_ts, pred_ts):.2f}')
    plt.show()


def exploratory_arima(ts, covariates=None, plot=False):

    if covariates:
        cov_scaler = Scaler()
        covariates = cov_scaler.fit_transform(covariates)

    AutoARIMA(trace=True, stepwise=True, information_criterion='bic').fit(ts,future_covariates=covariates)

    from darts.models import ARIMA
    arima = ARIMA(p=2, d=1, q=1, trend=None).fit(ts,future_covariates=covariates)
    arima_sm = arima.__getattribute__('model')

    if plot:
        arima_sm.plot_diagnostics(figsize=(15, 5))
        plt.show()

    if covariates:
        for idx, cov in enumerate(covariates.columns):
            print(f'x{idx + 1}', cov)

    print(arima_sm.summary())


def backtest_minute_data(model, ts_value, data_df, valid_from,
                         future_covariates=None, past_covariates=None,
                         stride = 1, forecast_horizon=2, last_points_only = True,
                         retrain=True, verbose=True,
                         scaler=None, plot=True):

    backtest = model.historical_forecasts(series=ts_value,
                                          future_covariates=future_covariates,
                                          past_covariates=past_covariates,
                                          start=pd.to_datetime(valid_from),
                                          retrain=retrain,
                                          verbose=verbose,
                                          forecast_horizon=forecast_horizon,
                                          stride=stride,
                                          last_points_only=last_points_only
                                          )
    if isinstance(backtest, list):
        print("Backtest is a list of forecasts - merging")
        days_ahead = []
        for bt in backtest:
            second_day = bt.time_index.date[-1]
            day_ahead = bt[pd.to_datetime(second_day):]
            days_ahead.append(day_ahead)

        backtest = timeseries.concatenate(days_ahead)

    if scaler:
        ts_value = scaler.inverse_transform(ts_value)
        backtest = scaler.inverse_transform(backtest)

    y_true = ts_value[backtest.time_index]['Value']
    y_pred = backtest
    resid_val = (y_true - y_pred).pd_series()
    tsplot(resid_val, label="Residuals on hold out")

    '''
    forward fill the dataframe with predicted values, 
    i.e. hourly is forward filled to all 15 minute step
    '''
    bt_df = backtest.pd_dataframe()
    data_eval = data_df.copy()
    data_eval.loc[bt_df.index, 'Value_pred'] = bt_df['Value']
    data_eval.loc[:, 'Value_pred'] = data_eval['Value_pred'].ffill()

    ts_eval = TimeSeries.from_dataframe(data_eval[['Value', 'Value_pred']])
    ts_eval = ts_eval[backtest.time_index[0]:]
    y_true = ts_eval['Value']
    y_pred = ts_eval['Value_pred']
    rmse_val = rmse(y_true, y_pred)
    mae_val = mae(y_true, y_pred)
    smape_val = smape(y_true, y_pred)


    resid_val = (y_true - y_pred).pd_series()
    plot_residuals(resid_val, title="15 minute residuals on hold out")

    if plot:
        fig = plt.figure(figsize=(12, 4))
        ts_eval['Value'].plot()
        ts_eval['Value_pred'].plot(label=f'{model.__class__.__name__} (steps={forecast_horizon})')
        plt.show()

    print(f"{model.__class__.__name__} n_val={len(ts_eval['Value'])} SMAPE={smape_val:.2f} RMSE={rmse_val:.2f} MAE={mae_val:.2f}")
    return backtest, ts_eval, rmse_val, mae_val, smape_val