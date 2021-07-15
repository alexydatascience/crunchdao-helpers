import pandas as pd


def ternarize_trend(dataframe):
    df = dataframe.copy()
    for col in df:
        df[col] = df[col] - df[col].shift().fillna(0)
        df[col] = df[col].apply(lambda x: -1 if x < 0 else 1 if x > 0 else 0)
    return df


def make_features(data, lags=1):
    X = data.copy()
    for i in range(X.Moons.max() + 1):
        feature_cols = [x for x in X.columns if 'feature' in x]
        for j in range(1, len(feature_cols)):
            X.loc[X.Moons == i, 'mean'] = X[X.Moons == 1][f'feature_{j}'].mean()
    X['sum'] = X[feature_cols].sum(axis=1)
    shifted_data = [X]
    for i in range(1, lags + 1):
        prev = X.shift(i)
        prev.columns = ['-1_' + x for x in X.columns]
        shifted_data.append(prev)
    out = pd.concat(shifted_data, axis=1).fillna(0)
    return out
