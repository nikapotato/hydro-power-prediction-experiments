# ---
# jupyter:
#   jupytext:
#     notebook_metadata_filter: ploomber
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
#   ploomber:
#     injected_manually: true
# ---

# %%
# Uncomment the next two lines to enable auto reloading for imported modules
# # %load_ext autoreload
# # %autoreload 2
# For more info, see:
# https://docs.ploomber.io/en/latest/user-guide/faq_index.html#auto-reloading-code-in-jupyter

# %% tags=["parameters"]
# If this task has dependencies, declare them in the YAML spec and leave this
# as None
from hydro_timeseries.util import load_timeseries_csv
import pandas as pd
from datetime import date

upstream = None

# This is a placeholder, leave it as None
product = None


# %% tags=["injected-parameters"]
# Parameters
valid_from = "2021-11-01"
random_seed = 1
upstream = {"load-data": {"nb": "../reports/load_data.ipynb", "data": "../data/data_raw.csv"}}
product = {"nb": "../reports/run_tests.ipynb", "data": "../data/data_raw_tested.csv"}


# %%
data = load_timeseries_csv(upstream['load-data']['data'])
data.to_csv(product['data'])

#TODO
'''
tests for consistency
- no gaps in training data
- no nans in crucial measurements
- etc
'''

# val = pd.read_csv(upstream['load-data']['data'], parse_dates=['Date_Time', 'Date'], index_col='Date_Time')
# val.head()

# %%
# assert val.iloc[0].Date.date() == date.fromisoformat(valid_from), "Validation set does not start from specified validation date"

# %%

