# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
# Add description here
#
# *Note:* You can open this file as a notebook (JupyterLab: right-click on it in the side bar -> Open With -> Notebook)


# %%
# Uncomment the next two lines to enable auto reloading for imported modules
# # %load_ext autoreload
# # %autoreload 2
# For more info, see:
# https://docs.ploomber.io/en/latest/user-guide/faq_index.html#auto-reloading-code-in-jupyter

# %% tags=["parameters"]
# If this task has dependencies, declare them in the YAML spec and leave this
# as None
from datetime import timedelta

import pandas as pd
from darts.models import RNNModel

from hydro_timeseries.util import load_timeseries_csv, add_mean_vars

upstream = None

# This is a placeholder, leave it as None
product = None


# %% tags=["injected-parameters"]
# Parameters
random_seed = 1
valid_from = "2021-11-01"
upstream = {"run-tests": {"nb": "../run_tests.ipynb", "data": "../data_raw_tested.csv"}}
product = {"nb": "../exploratory_darts_hourly.ipynb"}


# %%
data = load_timeseries_csv(upstream['run-tests']['data'])
data = add_mean_vars(data)
data = data.dropna(axis = 0)
valid_from = pd.to_datetime(valid_from)
train_until = valid_from - timedelta(days=1)

#%%
data_hourly = data.resample('h').mean()
train_hourly = data_hourly[:valid_from]
#%%
future_cov = data_hourly[['precip_mean', 'pressure_mean']]

'''
Generally speaking, `training_length` should have a higher value than `input_chunk_length`
because otherwise during training the RNN is never run for as many iterations as it will during
training.
'''

rnn_rain = RNNModel(input_chunk_length=20,
                    training_length=30,
                    model='GRU',
                    n_rnn_layers=2)

rnn_rain.fit(data_hourly['Value'],
             future_covariates=future_cov,
             epochs=400,
             verbose=True)

