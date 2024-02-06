from typing import Any

import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from matplotlib.ticker import MaxNLocator
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import STL
import warnings
from hydro_timeseries.util import smape, rmse, mae

warnings.simplefilter(action='ignore', category=FutureWarning)

def get_windows(y, cv):
    """Generate windows"""
    train_windows = []
    test_windows = []
    for i, (train, test) in enumerate(cv.split(y)):
        train_windows.append(train)
        test_windows.append(test)
    return train_windows, test_windows

def plot_windows(y, train_windows, test_windows, title=""):
    """Visualize training and test windows"""

    warnings.simplefilter("ignore", category=UserWarning)

    def get_y(length, split):
        # Create a constant vector based on the split for y-axis."""
        return np.ones(length) * split

    n_splits = len(train_windows)
    n_timepoints = len(y)
    len_test = len(test_windows[0])

    train_color, test_color = sns.color_palette("colorblind")[:2]

    fig, ax = plt.subplots(figsize=plt.figaspect(0.3))

    for i in range(n_splits):
        train = train_windows[i]
        test = test_windows[i]

        ax.plot(
            np.arange(n_timepoints), get_y(n_timepoints, i), marker="o", c="lightgray"
        )
        ax.plot(
            train,
            get_y(len(train), i),
            marker="o",
            c=train_color,
            label="Window",
        )
        ax.plot(
            test,
            get_y(len_test, i),
            marker="o",
            c=test_color,
            label="Forecasting horizon",
        )
    ax.invert_yaxis()
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set(
        title=title,
        ylabel="Window number",
        xlabel="Time",
        xticklabels=y.asfreq('D').index.date,
    )
    # remove duplicate labels/handles
    handles, labels = [(leg[:2]) for leg in ax.get_legend_handles_labels()]
    ax.legend(handles, labels)
    plt.show()



def plot_fea_importance(model, train_vars, file_name ='manual_feature_importance'):
    feature_imp = pd.DataFrame(sorted(zip(model.feature_importances_, train_vars.columns)),
                               columns=['Value', 'Feature'])

    plt.figure(figsize=(20, 10))
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False))
    plt.title(f'{model.__class__.__name__} Feature importances')
    plt.tight_layout()
    plt.savefig(f'plots/{file_name}.png')
    plt.show()


def plot_residuals(residuals, title = 'Kernel density of residuals'):
    """
        Density plot of residuals (y_true - y_pred) for testation set for given model
    """
    sns.set_palette(sns.color_palette("bright", 8))
    ax = sns.distplot(residuals, hist=True, kde=True,
                      kde_kws={'shade': True, 'linewidth': 3}, axlabel="Residual")
    title = ax.set_title(title)
    plt.tight_layout()
    plt.show()

def tsplot_pred(y_true, y_pred, label = 'Time series'):
    if not isinstance(y_true, pd.Series) or not isinstance(y_pred, pd.Series):
        y_true = pd.Series(y_true)
        y_pred = pd.Series(y_pred)

    fig = plt.figure(figsize=(14, 4))
    y_pred.plot(color='green')
    y_true.plot(color='blue')
    plt.show()



def tsplot(y, lags=10, label = 'Time series', y_pred = None, y_detr = None, do_plot_acf = True, filepath=None):
    if not isinstance(y, pd.Series):
        y = pd.Series(y)

    if y_pred is not None and not isinstance(y_pred, pd.Series):
        y_pred = pd.Series(y_pred)

    # layout
    if do_plot_acf:
        fig = plt.figure(figsize=(14, 6))
        ts_ax = plt.gca()
        layout = (2, 2)
        ts_ax = plt.subplot2grid(layout, (0, 0), colspan=2)
        acf_ax = plt.subplot2grid(layout, (1, 0))
        pacf_ax = plt.subplot2grid(layout, (1, 1))
    else:
        fig = plt.figure(figsize=(14, 5))
        ts_ax = plt.gca()

    y.plot(ax=ts_ax, color='blue')

    if y_pred is not None:
        y_pred.plot(ax=ts_ax, color='red', linewidth=2.0)

    if y_detr is not None:
        y_detr.plot(ax=ts_ax, linestyle='dashed', color='green', linewidth=0.5)

    ts_ax.set_title(label)

    # acf, pacf
    if do_plot_acf:
        plot_acf(y, lags=lags, ax=acf_ax, alpha=0.5, color='blue')
        plot_pacf(y, lags=lags, ax=pacf_ax, alpha=0.5, color='blue')

    plt.tight_layout()
    if filepath:
        plt.savefig(filepath)

    plt.show()

def trend_eval_plot(y_true: pd.Series, y_pred: pd.Series, label: str, compare: bool = False, filepath: Any = None):
    eval_df = pd.concat([y_true.rename('y_true'), y_pred.rename('y_pred')], axis=1)
    eval_df = eval_df.dropna(axis=0)
    smape_all = smape(eval_df['y_true'], eval_df['y_pred'])
    rmse_all = rmse(eval_df['y_true'], eval_df['y_pred'])
    mae_all = mae(eval_df['y_true'], eval_df['y_pred'])
    if compare:
        tsplot(eval_df['y_true'], do_plot_acf=False, y_pred=eval_df['y_pred'],
               label=f"{label} SMAPE={smape_all:.4f} RMSE={rmse_all:.2f} MAE={mae_all:.2f}", filepath=filepath)
    else:
        tsplot(eval_df['y_pred'], do_plot_acf=False,
               label=f"{label} SMAPE={smape_all:.4f} RMSE={rmse_all:.2f} MAE={mae_all:.2f}", filepath=filepath)


def seasonal_decompose_plot(series, period = 7, model='additive'):
    '''
    All time series data can be broken down into four core components:
    the average value,
    a trend (i.e. an increasing mean),
    seasonality (i.e. a repeating cyclical pattern),
    and a residual (random noise).

    model:
    If the seasonality’s amplitude is independent of the level then you should use the additive model,
    and if the seasonality’s amplitude is dependent on the level then you should use the multiplicative model.


    '''
    if period:
        result = seasonal_decompose(series, period=period, model=model)
    else:
        result = seasonal_decompose(series, model=model)

    # plt.figure(figsize=(15, 5))
    # plt.title(f"Seasonal decompose period={period} model={model}")
    fig = result.plot()
    fig.set_size_inches((15, 5))
    fig.tight_layout()
    plt.show()

    return result.trend, result.seasonal, result.resid




def stl_decompose_plot(series, period = 96):
    result = STL(series, period=period).fit()

    sns.set_palette(sns.color_palette("bright", 6))
    fig = result.plot()
    fig.set_size_inches((15,5))
    fig.tight_layout()
    plt.show()

    return result.trend, result.seasonal, result.resid




def plot_corr_matrix(corr, label="corrmat"):
    # Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(11, 9))

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(230, 20, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr, mask=mask, cmap=cmap, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5})

    plt.title(label)
    plt.tight_layout()
    plt.show()
