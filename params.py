import pandas as pd
from sklearn.feature_selection import SelectKBest, f_regression
from skopt.space import Categorical, Integer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import Ridge


max_dim = pd.read_csv('../data/X_train.csv').shape[1]

filter_features = {
    'reduce_dim': Categorical([SelectKBest(f_regression)]),
    'reduce_dim__k': Integer(5, max_dim)
}

scaler = {
    'scaler': Categorical([StandardScaler(), MinMaxScaler()])
}

ridge_search = {
    'model': Categorical([Ridge()]),
    'model__alpha': Integer(0, 1_000_000),
    'model__solver': Categorical(
        ['svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga']),
    'model__fit_intercept': Categorical([True, False]),
}
