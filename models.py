import itertools

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor


class RGBRegressor(BaseEstimator, RegressorMixin):
    """Returns predictions learned with dummy target.

    Parameters
    ----------
    dummy_target : pd.Series
        Target to fit on.
    estimator : estimator instance, default=LinearRegression()
        Estimator instance to fit.
    fit : bool, default=True
        If an estimator needs to be fitted.

    Attributes
    ----------
    estimator_ : estimator instance
        Fitted estimator
    """

    def __init__(self, dummy_target=None, estimator=None, fit_estimator=True):
        super().__init__()
        self.dummy_target = dummy_target
        if estimator is None:
            self.estimator = Ridge()
        else:
            self.estimator = estimator
        self.fit_estimator = fit_estimator

    def fit(self, X, y):
        if self.fit_estimator:
            self.estimator_ = self.estimator.fit(X, self.dummy_target[X.index])
        else:
            self.estimator_ = self.estimator
        return self

    def predict(self, X):
        return self.estimator_.predict(X)


class TargetsRegressor(BaseEstimator, RegressorMixin):
    """Learns from the other targets in multioutput case.

    Parameters
    ----------
    ys : pd.DataFrame
        Targets to fit on.
    base_estimators : list of estimator instances, default=LinearRegression()
        Estimator instances to fit.
    final_estimator : estimator instance
        An estimator to fit on `base_estimators`' predictions.
    use_features : bool, default=False
        Whether to use original X along with `ys` to fit `final_estimator`.
    """

    def __init__(self, ys=None, base_estimators=None,
                 final_estimator=None, use_features=False):
        super().__init__()
        self.ys = ys
        if base_estimators is None:
            self.base_estimators = [Ridge()] * ys.shape[1]
        else:
            assert ys.shape[1] == len(base_estimators)
            self.base_estimators = base_estimators
        if final_estimator is None:
            self.final_estimator = Ridge()
        else:
            self.final_estimator = final_estimator
        self.use_features = use_features

    def fit(self, X, y):
        for i, model in enumerate(self.base_estimators):
            model.fit(X, self.ys.iloc[X.index, i])
        if self.use_features:
            X_ = pd.concat([X, self.ys.iloc[X.index]], axis=1)
        else:
            X_ = self.ys.iloc[X.index]
        self.final_estimator.fit(X_, y)
        return self

    def predict(self, X):
        base_preds = [pd.Series(m.predict(X)) for m in self.base_estimators]
        base_preds = pd.concat(base_preds, axis=1).set_index(X.index)
        if self.use_features:
            X_ = pd.concat([X, base_preds], axis=1)
        else:
            X_ = base_preds
        y_pred = self.final_estimator.predict(X_)
        return y_pred


class RegressorsChain(BaseEstimator, RegressorMixin):
    def __init__(self, base_estimators=None, final_estimators=None, cascade_depth=None):
        super().__init__()
        self.base_estimators = base_estimators
        self.final_estimators = final_estimators
        self.cascade_depth = cascade_depth

    def fit(self, X, y):
        if self.base_estimators is None:
            self.base_estimators = [Ridge()] * y.shape[1]
        if self.final_estimators is None:
            self.final_estimators = [Ridge()] * y.shape[1]
        for i, model in enumerate(self.base_estimators):
            model.fit(X, y.iloc[:, i])
        extended_features = self.extend_features(X, y)
        for model, (X_ext, y_res) in zip(self.final_estimators, extended_features):
            model.fit(X_ext, y_res)
        return self

    def predict(self, X):
        be_preds = self.base_estimators_predict(X)
        fe_preds = []
        n_targets = len(be_preds)
        combinations = self.combinationsS(n_targets, n_targets - 1)
        for fe, comb in zip(self.final_estimators, combinations):
            X_ext = np.concatenate((X, *(be_preds[i] for i in comb)), axis=1)
            pred = fe.predict(X_ext)
            fe_preds.append(pred)
        pred = np.concatenate(fe_preds, axis=1)
        if self.cascade_depth is not None:
            return self.cascade(X, pred, depth=self.cascade_depth)
        return pred

    def combinationsS(self, n, k):
        combinations = list(itertools.combinations(range(n), k))
        return reversed(combinations)

    def extend_features(self, X, y):
        X, y = np.array(X), np.array(y)
        n_targets = y.shape[1]
        combinations = self.combinationsS(n_targets, n_targets - 1)
        for i, comb in enumerate(combinations):
            X_ = np.concatenate((X, y[:, comb]), axis=1)
            y_ = y[:, [i]]
            yield X_, y_

    def base_estimators_predict(self, X):
        be_preds = []
        for be in self.base_estimators:
            pred = be.predict(X).reshape(-1, 1)
            be_preds.append(pred)
        return be_preds

    def cascade(self, X, y, depth, count=0):
        if depth == count:
            return y
        n_targets = y.shape[1]
        combinations = self.combinationsS(n_targets, n_targets - 1)
        for i, (comb, fe) in enumerate(zip(combinations, self.final_estimators)):
            X_ = np.concatenate((X, y[:, comb]), axis=1)
            y[:, [i]] = fe.predict(X_).reshape(-1, 1)
        return self.cascade(X, y, depth, count + 1)


def combinations_(n, k):
    combinations = list(itertools.combinations(range(n), k))
    return reversed(combinations)


def cascade(final_estimators, X, y, depth=1, count=0):
    if depth == count:
        return y
    n_targets = y.shape[1]
    combinations = combinations_(n_targets, n_targets - 1)
    for i, (comb, fe) in enumerate(zip(combinations, final_estimators)):
        X_ = np.concatenate((X, y[:, comb]), axis=1)
        y[:, [i]] = fe.predict(X_).reshape(-1, 1)
    return cascade(final_estimators, X, y, depth, count + 1)


# class RegressorsChain(BaseEstimator, RegressorMixin):
#     def __init__(self, model1=None, model2=None, ids=None):
#         """
#         model1: base model to predict tagret(s) in according to ids
#         model2: model to predict remaining target(s).
#         ids: indices to be added to X
#         """
#         super().__init__()
#         self.model1 = model1
#         self.model2 = model2
#         self.ids = ids
#
#     def predict(self, X):
#         X = np.array(X)
#         model1_pred = self.model1.predict(X)
#         X_ext = np.concatenate([X, model1_pred], axis=1)
#         model2_pred = self.model2.predict(X_ext)
#         concat_pred = np.concatenate([model1_pred, model2_pred], axis=1)
#         mixed_ids = self.ids + self.y_res_ids_
#
#         res = np.zeros_like(concat_pred)
#         for i, j in enumerate(mixed_ids):
#             res[:, j] = concat_pred[:, i]
#
#         # remixed_rows = [concat_pred[:, i].reshape(-1, 1) for i in mixed_ids]
#         return res  # np.concatenate(remixed_rows, axis=1)
#
#     def fit(self, X, y):
#         X, y = np.array(X), np.array(y)
#         self.model1 = MultiOutputRegressor(self.model1)
#         self.model2 = MultiOutputRegressor(self.model2)
#         self.targets_ids_ = range(y.shape[1])
#         y_cut = y[:, self.ids]
#         self.model1.fit(X, y_cut)
#         X_ext = np.concatenate([X, y_cut], axis=1)
#         self.y_res_ids_ = [i for i in self.targets_ids_ if i not in self.ids]
#         y_res = y[:, self.y_res_ids_]
#         self.model2.fit(X_ext, y_res)
#         return self
#
#     def set_params(self, **params):
#         if not params:
#             return self


class RegressorsConcat(BaseEstimator, RegressorMixin):
    def __init__(self, *models):
        """
        models: tuples like (estimator, indices), where 'indices' is
        an array of one or more target's indices to be predicted with the estimator
        """
        self.models = models

    def predict(self, X):
        X = np.array(X)
        preds = []
        for model in self.fitted_models_:
            pred = model.predict(X)
            if len(pred.shape) == 1:
                pred = pred.reshape(-1, 1)
                print(pred.shape)
            preds.append(pred)
        concat_pred = np.concatenate(preds, axis=1)
        res = np.zeros_like(concat_pred)
        for i, j in enumerate(self.ids_):
            res[:, j] = concat_pred[:, i]
        return res

    def fit(self, X, y):
        X, y = np.array(X), np.array(y)
        self.fitted_models_, ids = [], []
        for model, i in self.models:
            model = MultiOutputRegressor(model)
            y_i = y[:, i]
            if len(y_i.shape) == 1:
                y_i = y_i.reshape(-1, 1)
            fitted = model.fit(X, y_i)
            self.fitted_models_.append(fitted)
            ids.append(i)
        self.ids_ = [z for y in \
                     (x if isinstance(x, tuple) else [x] for x in ids) \
                     for z in y]
        return self
