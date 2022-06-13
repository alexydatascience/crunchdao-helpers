import pickle
from collections import OrderedDict

import numpy as np
import pandas as pd
import requests
import os

def pred_from_dictm(X_test, params, pred_suffix, save_csv=True):
    """Returns prediction DataFrame from dict of models
    (e.g. dcl.utils.dict_rgbmodels('params_2'))
    """
    models = dict_rgbmodels(params)
    preds = [m.predict(X_test) for m in models.values()]
    pred_df = rgb_df(preds)
    if save_csv:
        pred_df.to_csv(f'pred_{pred_suffix}.csv', index=False)
    return pred_df


def load_models(*model_names):
    return (pickle.load(open(name, 'rb')) for name in model_names)


def dict_rgbmodels(params):
    """
    params should be taken from name like target_{}_params.save,
    e.g. target_r_stack_params_1_2.sav --> stack_params_1_2
    """
    models_dict = OrderedDict()
    for y in ['r', 'g', 'b']:
        name = f'target_{y}_{params}.sav'
        models_dict[y] = pickle.load(open(name, 'rb'))
    return models_dict


def rgb_df(preds):
    if isinstance(preds, np.ndarray):
        r_pred, g_pred, b_pred = [row for row in preds.T]
    if isinstance(preds, list):
        r_pred, g_pred, b_pred = preds
    prediction = pd.DataFrame({
        'target_r': np.array(r_pred).flatten(),
        'target_g': np.array(g_pred).flatten(),
        'target_b': np.array(b_pred).flatten()
    })
    return prediction


def softmax(x):
    return np.exp(x) / sum(np.exp(x))


def submit(prediction):
    if not isinstance(prediction, pd.DataFrame):
        prediction = pd.DataFrame(
            prediction, columns=['target_r', 'target_g', 'target_b'])

    if any(prediction.max() > 1) or any(prediction.min() < 0):
        prediction = prediction.apply(softmax)
    api_key = os.environ.get("API_KEY")

    r = requests.post("https://tournament.datacrunch.com/api/submission",
                      files={
                          "file": ("x", prediction.to_csv().encode('ascii'))
                      },
                      data={
                          "apiKey": api_key
                      },
                      )

    if r.status_code == 200:
        print("Submission submitted :)")
    elif r.status_code == 423:
        print("ERR: Submissions are close")
        print("You can only submit during rounds eg: Friday 7pm GMT+1 to Sunday midnight GMT+1.")
        print("Or the server is currently crunching the submitted files, please wait some time before retrying.")
    elif r.status_code == 422:
        print("ERR: API Key is missing or empty")
        print("Did you forget to fill the API_KEY variable?")
    elif r.status_code == 404:
        print("ERR: Unknown API Key")
        print(
            "You should check that the provided API key is valid and is the same as the one you've received by email.")
    elif r.status_code == 400:
        print("ERR: The file must not be empty")
        print("You have send a empty file.")
    elif r.status_code == 401:
        print("ERR: Your email hasn't been verified")
        print("Please verify your email or contact a cruncher.")
    elif r.status_code == 429:
        print("ERR: Too many submissions")
    else:
        print("ERR: Server returned: " + str(r.status_code))
        print(
            "Ouch! It seems that we were not expecting this kind of result from the server, if the probleme persist, contact a cruncher.")
