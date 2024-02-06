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

# %% tags=["parameters"]
# If this task has dependencies, declare them in the YAML spec and leave this
# as None
upstream = None
# This is a placeholder, leave it as None
product = None
random_seed = None
valid_from = None
file_path = None
# %% tags=["injected-parameters"]
# Parameters
# file_path = "/hydro-power-prediction/data/data_mve_production_forecast_test.csv"
random_seed = 1
valid_from = "2021-11-01"
product = {"nb": "../load_data.ipynb", "data": "../data_raw.csv"}


# %%
from hydro_timeseries.util import load_timeseries_csv

# %%
'''
Load all data into the pipeline
'''
data = load_timeseries_csv(file_path)
# data = pd.read_csv(file_path, parse_dates=['Date_Time', 'Date'], index_col='Date_Time')
data.drop(columns=['Date'], inplace = True)

data.to_csv(product['data'])
