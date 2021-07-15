from collections import OrderedDict

import numpy as np
import pandas as pd
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from metrics import scorer


def cross_val_targets(model, X, y, scale=False, cv=3, scoring=None):
    """
    Returns
    -------
    Array of scores of the estimator for each run of the cross validation.
    """
    if not isinstance(y, pd.DataFrame):
        y = pd.DataFrame(y)
    if scale:
        pipe = make_pipeline(StandardScaler(), model)
    else:
        pipe = model
    if scoring is None:
        scoring = make_scorer(scorer)
    scores = OrderedDict()
    for i in range(y.shape[1]):
        score = cross_val_score(pipe, X, np.array(y)[:, i], scoring=scoring, cv=cv)
        scores[str(y.columns[i])] = np.append(score, score.mean())
    out = pd.DataFrame.from_dict(scores, orient='index')
    out.columns = [f'cv_{i + 1}' for i in range(cv)] + ['mean']
    return out
