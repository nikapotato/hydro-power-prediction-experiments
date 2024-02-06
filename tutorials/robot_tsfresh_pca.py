from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd


class PCAForPandas(PCA):
    """This class is just a small wrapper around the PCA estimator of sklearn including normalization to make it
    compatible with pandas DataFrames.
    """

    def __init__(self, **kwargs):
        self._z_scaler = StandardScaler()
        super(self.__class__, self).__init__(**kwargs)

        self._X_columns = None

    def fit(self, X, y=None):
        """Normalize X and call the fit method of the base class with numpy arrays instead of pandas data frames."""

        X = self._prepare(X)

        self._z_scaler.fit(X.values, y)
        z_data = self._z_scaler.transform(X.values, y)

        return super(self.__class__, self).fit(z_data, y)

    def fit_transform(self, X, y=None):
        """Call the fit and the transform method of this class."""

        X = self._prepare(X)

        self.fit(X, y)
        return self.transform(X, y)

    def transform(self, X, y=None):
        """Normalize X and call the transform method of the base class with numpy arrays instead of pandas data frames."""

        X = self._prepare(X)

        z_data = self._z_scaler.transform(X.values, y)

        transformed_ndarray = super(self.__class__, self).transform(z_data)

        pandas_df = pd.DataFrame(transformed_ndarray)
        pandas_df.columns = ["pca_{}".format(i) for i in range(len(pandas_df.columns))]

        return pandas_df

    def _prepare(self, X):
        """Check if the data is a pandas DataFrame and sorts the column names.

        :raise AttributeError: if pandas is not a DataFrame or the columns of the new X is not compatible with the
                               columns from the previous X data
        """
        if not isinstance(X, pd.DataFrame):
            raise AttributeError("X is not a pandas DataFrame")

        X.sort_index(axis=1, inplace=True)

        if self._X_columns is not None:
            if self._X_columns != list(X.columns):
                raise AttributeError("The columns of the new X is not compatible with the columns from the previous X data")
        else:
            self._X_columns = list(X.columns)

        return X

# %%
from tsfresh.examples.robot_execution_failures import download_robot_execution_failures, load_robot_execution_failures
from tsfresh.feature_extraction import extract_features
from tsfresh.feature_selection import select_features
from tsfresh.utilities.dataframe_functions import impute
from tsfresh.feature_extraction import ComprehensiveFCParameters, MinimalFCParameters, settings

download_robot_execution_failures()
df, y = load_robot_execution_failures()
df_train = df.iloc[(df.id <= 87).values]
y_train = y[0:-1]

df_test = df.iloc[(df.id >= 87).values]
y_test = y[-2:]

df.head()
# %%
X_train = extract_features(df_train, column_id='id', column_sort='time', default_fc_parameters=MinimalFCParameters(),
                           impute_function=impute)

# %%
X_train.head()

# %%
X_train_filtered = select_features(X_train, y_train)
X_train_filtered.tail()

# %%
pca_train = PCAForPandas(n_components=4)
X_train_pca = pca_train.fit_transform(X_train_filtered)

# add index plus 1 to keep original index from robot example
X_train_pca.index += 1

X_train_pca.tail()

# %%
X_test_filtered = extract_features(df_test, column_id='id', column_sort='time',
                                   kind_to_fc_parameters=settings.from_columns(X_train_filtered.columns),
                                   impute_function=impute)

X_test_filtered

# %%
X_test_pca = pca_train.transform(X_test_filtered)

# reset index to keep original index from robot example
X_test_pca.index = [87, 88]

X_test_pca
