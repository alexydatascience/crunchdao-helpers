import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon


def check_jsd(X, X_test, n_samples=100, return_lst=False):
    X_ids = np.random.choice(X.index, n_samples)
    X_test_ids = np.random.choice(X_test.index, n_samples)
    js_lst = []
    for i, j in zip(X_ids, X_test_ids):
        js = jensenshannon(X.loc[i], X_test.loc[j])
        js_lst.append(js)
    if return_lst:
        return js_lst
    return pd.DataFrame(js_lst).describe().transpose()


def jsd_ids(X, X_test, threshold, n=2, n_samples=None):
    """Returns indices of n X rows closest to every X_test row
       by threshold (Jensen-Shannon Divergence value)
    ---
    Returns : ids, js_lst, no_pair_ids
    """
    ids, js_lst, no_pair_ids = [], [], []
    if n_samples is None:
        n_samples = X.shape[0]
    for i in X_test.index:
        X_ids_sample = np.random.choice(X.index, n_samples)
        cnt = 0
        for j in X_ids_sample:
            if j not in ids:
                js = jensenshannon(X_test.loc[i], X.loc[j])
                if js < threshold:
                    ids.append(j)
                    js_lst.append(js)
                    cnt += 1
                    if cnt == n:
                        break
            if j == X_ids_sample[-1] and cnt == 0:
                no_pair_ids.append(i)
                print(f'X_test.loc[{i}] has no pair')
    return ids, js_lst, list(set(no_pair_ids))
