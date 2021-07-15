import itertools
from collections import OrderedDict

import pandas as pd
from sklearn.feature_selection import RFECV
from sklearn.linear_model import Ridge
from sklearn.metrics import make_scorer

from cv import cross_val_targets
from metrics import scorer


def select_features_(X, y, model, cv, scoring, min_features_to_select, scale=False):
    """
    RFECV selection for a single target
    """
    selector = RFECV(
        model, cv=cv, scoring=scoring, min_features_to_select=min_features_to_select)
    selector = selector.fit(X, y)
    support = selector.support_
    df = cross_val_targets(
        model, X.iloc[:, support], y, scale=scale, cv=cv, scoring=scoring)
    df['n_features'] = selector.support_.sum()
    return df, selector.estimator_, support


def select_features(X, y, model=None, times=None, cv=3,
                    scoring=make_scorer(scorer), scale=False):
    if model is None:
        model = Ridge()
    if not isinstance(y, pd.DataFrame):
        y = pd.DataFrame(y)
    support_dict = OrderedDict()
    models_dict = OrderedDict()
    for col in y:
        res = []
        if times is None:
            df, estimator, support = select_features_(
                X, y[col], model, cv, scoring, min_features_to_select=1, scale=scale)
            res.append(df)
            support_dict[col] = support
            models_dict[col] = estimator
            display(pd.concat(res))
        else:
            for i in range(1, times):
                df, estimator, support = select_features_(
                    X, y[col], model, cv, scoring, min_features_to_select=i)
                res.append(df)
            display(pd.concat(res))
    return models_dict, support_dict


def get_combinations(n, k):
    combinations = list(itertools.combinations(range(n), k))
    return reversed(combinations)


def select_features_ext(X, y, model=None, cv=3, scoring=make_scorer(scorer), scale=False):
    if model is None:
        model = Ridge()
    n_targets = y.shape[1]
    combinations = get_combinations(n_targets, n_targets - 1)
    support_dict = OrderedDict()
    models_dict = OrderedDict()
    for comb in combinations:
        y_res_ids = list(set(range(n_targets)) - set(comb))
        y_res = y.iloc[:, y_res_ids]
        y_to_x = y.iloc[:, list(comb)]
        X_ext = pd.concat([X, y_to_x], axis=1)
        selector = RFECV(model, cv=cv, scoring=scoring)
        selector.fit(X_ext, y_res)
        support = selector.support_
        support_dict[str(*y_res) + '_ext'] = support
        models_dict[str(*y_res) + '_ext'] = selector.estimator_
        df = cross_val_targets(
            model, X_ext.iloc[:, support], y_res, cv=cv, scoring=scoring, scale=scale)
        df['n_features'] = support.sum()
        display(df)
    return models_dict, support_dict
