import scipy.stats as st
import numpy as np


def scorer(y_test, y_pred, return_mean=True):
    assert len(y_test) == len(y_pred)
    y_test = np.array(y_test).reshape(-1, 1)
    y_pred = np.array(y_pred).reshape(-1, 1)
    scores = []
    for i in range(y_test.shape[1]):
        score = (st.spearmanr(y_test[:, i], y_pred[:, i]))[0] * 100
        scores.append(score)
    if return_mean:
        return np.mean(scores)
    return np.array(scores)
