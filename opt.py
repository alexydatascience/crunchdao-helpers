import pandas as pd
import pickle
import itertools
import os
from collections import defaultdict
from tqdm import tqdm
from sklearn.metrics import make_scorer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from skopt import BayesSearchCV

from .metrics import scorer


class Skoptimizer(object):
    def __init__(self, pipe=None, scoring=None, dir=None, random_state=None):
        if scoring is None:
            self.scoring = make_scorer(scorer)
        else:
            self.scoring = scoring
        if dir is None:
            os.makedirs('models', exist_ok=True)
            self.dir = 'models'
        else:
            self.dir = dir
        self.random_state = random_state
        if pipe is None:
            self.pipe = Pipeline([
                ('scaler', StandardScaler()),
                ('poly', 'passthrough'),
                ('reduce_dim', 'passthrough'),
                ('model', Ridge())
            ])
        else:
            self.pipe = pipe
        self.scores_ = []

    def bayes_fit(self, X, y, params, suffix='model', cv=3, X_test=None, y_test=None):
        if not isinstance(y, pd.DataFrame):
            y = pd.DataFrame(y)

        col = str(*y)
        opt_name = f'{self.dir}/{col}_{suffix}.sav'
        if os.path.exists(opt_name):
            print(f'\n{opt_name} already exists')
        else:
            opt = BayesSearchCV(
                self.pipe, params, cv=cv, scoring=self.scoring,
                                random_state=self.random_state)
            opt.fit(X, y)

            score = opt.best_score_
            self.scores_.append((suffix, col, score))
            pickle.dump(opt.best_estimator_, open(opt_name, 'wb'))

            print(f'\nval. score: {score}')
            if X_test is not None:
                print(f'test score: {opt.score(X_test, y_test)}')

            return opt.best_estimator_

    def get_scores(self):
        d = defaultdict(list)
        for model, target, score in self.scores_:
            d['model'].append(model)
            d[target].append(score)
        d['model'] = list(set(d['model']))
        return pd.DataFrame.from_dict(d)

    def bayes_fit_all(self, X, y, params, suffix='model', cv=3, X_test=None, y_test=None):
        for i in tqdm(range(y.shape[1])):
            yield self.bayes_fit(X, y.iloc[:, [i]], params=params, cv=cv,
                                 suffix=suffix, X_test=X_test, y_test=y_test)

    def combinations_(self, n, k):
        combinations = list(itertools.combinations(range(n), k))
        return reversed(combinations)

    def bayes_ext_fit(self, X, y, params, suffix='model', cv=3, X_test=None, y_test=None):
        print('Extended dataset search...')
        suffix = 'ext_' + suffix
        n_targets = y.shape[1]
        combinations = self.combinations_(n_targets, n_targets - 1)
        for comb in tqdm(combinations):
            y_res_ids = list(set(range(n_targets)) - set(comb))
            y_res = y.iloc[:, y_res_ids]
            y_to_x = y.iloc[:, list(comb)]
            X_ext = pd.concat([X, y_to_x], axis=1)
            yield self.bayes_fit(X_ext, y_res, params=params, cv=cv,
                                 suffix=suffix, X_test=X_test, y_test=y_test)

    def predict_all_(self, init_models, X_test, init_masks=None):
        im_preds = []
        if init_masks is None:
            init_masks = [[True] * X_test.shape[1]] * len(init_models)
        for im, mask in zip(init_models, init_masks):
            pred = im.predict(X_test.iloc[:, mask])
            pred = pd.DataFrame(pred)
            im_preds.append(pred)
        return im_preds

    def ext_predict(self, init_models, extended_models, X_test, return_im_preds=False,
                    init_masks=None, extended_masks=None):
        im_preds = self.predict_all_(init_models, X_test, init_masks)
        em_preds = []
        n_targets = len(im_preds)
        if extended_masks is None:
            extended_masks = [[True] * (X_test.shape[1] + n_targets - 1)] * len(extended_models)
        combinations = self.combinations_(n_targets, n_targets - 1)
        for em, mask, comb in zip(extended_models, extended_masks, combinations):
            X_test_ext = pd.concat([X_test, *(im_preds[i] for i in comb)], axis=1)
            pred = em.predict(X_test_ext.iloc[:, mask])
            em_preds.append(pred)
        if return_im_preds:
            return im_preds, em_preds
        else:
            return em_preds
