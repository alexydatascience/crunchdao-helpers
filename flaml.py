import pickle
from collections import OrderedDict

import numpy as np
import scipy.stats as st
from tqdm import tqdm
from flaml import AutoML


class RGBFLAML(object):
    """estimator_list: ['lgbm', 'xgboost', 'catboost', 'rf', 'extra_tree']
    """
    def __init__(self, time_budget=60, suffix='flaml', estimator_list='auto',
                 metric=None, task='regression', eval_method='cv', n_splits=3,
                 log_file_name='/content/flaml.log'):
        self.suffix = suffix
        if metric is None:
            metric = self.spearmanr_metric
        self.automl = AutoML()
        self.automl_settings = {
            'time_budget': time_budget,
            'metric': metric,
            'estimator_list': estimator_list,
            'task': task,
            'eval_method': eval_method,
            'n_splits': n_splits,
            'log_file_name': log_file_name,
        }

    def spearmanr_metric(self, X_test, y_test, estimator, labels, X_train, y_train,
                         weight_test=None, weight_train=None):
        y_pred = estimator.predict(X_test)
        test_loss = -(st.spearmanr(y_test, y_pred) * 100)[0]
        y_pred = estimator.predict(X_train)
        train_loss = -(st.spearmanr(y_train, y_pred) * 100)[0]
        return test_loss * 100, [test_loss * 100, train_loss * 100]

    def fit_one(self, X, y):
        self.automl.fit(X_train=X, y_train=np.array(y), **self.automl_settings)
        model_name = f'{str(*y)}_{self.suffix}.sav'
        pickle.dump(self.automl.model, open(model_name, 'wb'))
        pickle.dump(self.automl.model, open(f'{model_name}', 'wb'))
        return self.automl.model

    def fit_all(self, X, y):
        models_dict = OrderedDict()
        for i in tqdm(range(y.shape[1])):
            y_ = y.iloc[:, [i]]
            models_dict[str(*y_)] = self.fit_one(X, y_)
            print(type(y_))
        return models_dict
