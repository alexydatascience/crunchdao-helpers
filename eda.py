import itertools

import numpy as np
import pandas as pd
import scipy.stats as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import Ridge
from sklearn.metrics import make_scorer, matthews_corrcoef
from sklearn.model_selection import cross_val_score
import plotly.graph_objects as go

from metrics import scorer


def mid_range(a, axis=None):
    rng = a.max(axis=axis) - a.min(axis=axis)
    return rng / 2


def custom_std(a, outer_fn, inner_fn, axis=None):
    a = np.array(a)
    diffs = (a - inner_fn(a, axis=axis).reshape(-1, 1)) ** 2
    return outer_fn(diffs, axis=axis) ** 0.5


def mode(a, axis=None):
    return st.mode(a, axis=axis)[0]


def stat_features(df):
    res = df.copy()
    res['mean'] = np.mean(res, axis=1)
    res['median'] = np.median(res, axis=1)
    res['mode'], res['mode_n_val'] = st.mode(res, axis=1)
    res['mid_range'] = mid_range(res, axis=1)
    res['std'] = np.std(res, axis=1)
    fns = np.mean, np.median, mode, mid_range
    perm = itertools.permutations(fns, 2)
    for comb in perm:
        col_name = 'std_' + comb[0].__name__ + '_' + comb[1].__name__
        res[col_name] = custom_std(res, *comb, axis=1)
    return res


def plot_rgb_scores(X, y, reverse=False):
    X_sample, y_sample = X.copy(), y.copy()
    fig = go.Figure()
    if reverse:
        for col in X_sample.columns:
            X_sample[col] = list(reversed(X_sample[col]))
    for target in y_sample.columns:
        scores = [scorer(X_sample[col], y_sample[target]) for col in X_sample.columns]
        fig.add_trace(go.Scatter(x=X_sample.columns, y=scores, mode='lines', name=target))
    fig.update_layout(title='Spearman Correlation Between Features and Targets',
                      xaxis_title='Features',
                      yaxis_title='Spearmanr')
    fig.show()

def get_corr_features(df, threshold):
    lst = []
    for row in df.index:
        for col in df.columns:
            if df.loc[row, col] > threshold and row != col:
                if {(row, col)} not in lst:
                    lst.append({(row, col)})
    return list(set([list(x)[0] for x in lst]))


def pred_corr(pred_range=None, targets=None):
    """Spearmanr between saved predictions.
       Predictions must have names like pred_{number}.csv
       ---
       Displays heatmaps for every target and mean.
       Returns list of dataframes.
    """
    if pred_range is None:
        start, stop = (1, 12)
    elif isinstance(pred_range, tuple):
        start, stop = pred_range
    elif isinstance(pred_range, int):
        start, stop = (1, pred_range)
    if targets is None:
        targets = ['target_r', 'target_g', 'target_b']
    n = len(targets)
    dfs = []
    for i in range(n + 1):
        dfs.append(pd.DataFrame())
    for r, c in itertools.combinations(range(start, stop), 2): 
        preds = [pd.read_csv(f'pred_{i}.csv') for i in (r, c)]
        scores = [scorer(preds[0][t], preds[1][t]) for t in targets]
        for i in range(n):
            dfs[i].loc[r, c] = scores[i]
            dfs[i].index.name = targets[i]
        dfs[-1].loc[r, c] = np.mean(scores)
        dfs[-1].index.name = 'mean'
    fig, ax = plt.subplots(1, n + 1, figsize=(26, 5.5))
    for i in range(n + 1):
        sns.heatmap(dfs[i], xticklabels=dfs[i].columns, yticklabels=dfs[i].index, 
                    annot=True, fmt='.2f', cmap='viridis', ax=ax[i])
    plt.show()
    return dfs


class CovariateShiftClassifier(object):
    def __init__(self, estimator=None, n_samples=None,
                 threshold=0.2, random_state=None):
        self.estimator = estimator
        self.n_samples = n_samples
        self.threshold = threshold
        self.random_state = random_state

    def fit(self, X_train, X_test, return_comb=False):
        if self.n_samples is None:
            len_train, len_test = len(X_train), len(X_test)
            self.n_samples = len_test if len_train > len_test else len_train
        X_train['origin'] = 0
        X_test['origin'] = 1
        X_train = X_train.sample(self.n_samples, random_state=self.random_state)
        X_test = X_test.sample(self.n_samples, random_state=self.random_state)
        X = X_train.append(X_test)
        y = X['origin']
        X.drop('origin', axis=1, inplace=True)
        self.X, self.y = X, y
        if return_comb:
            return self.X, self.y
        return self

    def classify(self):
        if self.estimator is None:
            self.estimator = Ridge(random_state=self.random_state)
        drop_list = []
        scorer = make_scorer(matthews_corrcoef)
        for col in self.X.columns:
            X_col = pd.DataFrame(self.X[col])
            score = cross_val_score(
                self.estimator, X_col, self.y, cv=2, scoring=scorer)
            if (np.mean(score) > self.threshold):
                drop_list.append(col)
        return drop_list
