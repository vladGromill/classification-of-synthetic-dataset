import re
import os
import sys
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV

from sklearn.preprocessing import StandardScaler, MinMaxScaler
PROJECT_DIR = Path(os.getcwd())
sys.path.append(str(Path(PROJECT_DIR, 'utils')))
import make_full_pipeline
import importlib
importlib.reload(make_full_pipeline)
from make_full_pipeline import Pipeline_log_regression, PipelineCatBoostClassifier
from sklearn.model_selection import StratifiedKFold
from itertools import product


def custom_grid_search(X, y, pipeline, param_grid, cv_splits=5):
    """
    Собственный перебор параметров с расчетом logloss и roc_auc.

    Args:
    - X: numpy array, признаки.
    - y: numpy array, таргет.
    - pipeline: объект кастомного пайплайна.
    - param_grid: словарь, где ключи — параметры, а значения — список возможных значений.
    - cv_splits: количество фолдов для кросс-валидации.

    Returns:
    - results: отсортированный список словарей с параметрами и метриками.
    """
    results = []
    skf = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=42)

    keys, values = zip(*param_grid.items())
    all_combinations = [dict(zip(keys, v)) for v in product(*values)]
    i_par = 1
    for params in all_combinations:
        logloss_scores = []
        roc_auc_scores = []
        #i_fold = 1
        for train_idx, val_idx in skf.split(X, y):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            pipeline.set_params(**params)

            pipeline.fit(X_train, y_train)

            evall = pipeline.evaluate_metrics(X_val, y_val)
            logloss_cur, roc_auc_cur = evall['logloss'], evall['roc_auc']
            #y_pred_proba = pipeline.predict_proba(X_val)

            logloss_scores.append(logloss_cur)
            roc_auc_scores.append(roc_auc_cur)
            #print(f'Metrics on {i_par} iteration on {i_fold} fold: LogLoss: {logloss_cur}, roc auc: {roc_auc_cur}')

        mean_logloss = np.mean(logloss_scores)
        mean_roc_auc = np.mean(roc_auc_scores)
        #print(f'Mean metrics on {i_par} iteration: LogLoss: {mean_logloss}, roc auc: {mean_roc_auc}')
        results.append({
            "params": params,
            "logloss": mean_logloss,
            "roc_auc": mean_roc_auc
        })
        i_par += 1

    # Сортировка по logloss (чем меньше, тем лучше)
    results = sorted(results, key=lambda x: x["logloss"])
    return results

def custom_grid_search_catboost(X, y, pipeline, param_grid, cv_splits=2):
    """
    Собственный перебор параметров с расчетом logloss и roc_auc.

    Args:
    - X: numpy array, признаки.
    - y: numpy array, таргет.
    - pipeline: объект кастомного пайплайна.
    - param_grid: словарь, где ключи — параметры, а значения — список возможных значений.
    - cv_splits: количество фолдов для кросс-валидации.

    Returns:
    - results: отсортированный список словарей с параметрами и метриками.
    """
    results = []
    skf = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=42)

    keys, values = zip(*param_grid.items())
    all_combinations = [dict(zip(keys, v)) for v in product(*values)]
    i_par = 1
    for params in all_combinations:
        logloss_scores = []
        roc_auc_scores = []
        i_fold = 1
        for train_idx, val_idx in skf.split(X, y):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            pipeline.set_params(**params)

            pipeline.fit(X_train, y_train)

            evall1 = pipeline.evaluate_metrics(X_val, y_val, calibrate=True)
            evall2 = pipeline.evaluate_metrics(X_val, y_val, calibrate=False)
            logloss_cur1, roc_auc_cur1 = evall1['logloss'], evall1['roc_auc']
            logloss_cur2, roc_auc_cur2 = evall2['logloss'], evall2['roc_auc']
            #y_pred_proba = pipeline.predict_proba(X_val)
            if logloss_cur2 >= logloss_cur1:
                logloss_cur = logloss_cur1
                roc_auc_cur = roc_auc_cur1
                params['calibrate'] = True
            else:
                logloss_cur = logloss_cur2
                roc_auc_cur = roc_auc_cur2
                params['calibrate'] = False
            logloss_scores.append(logloss_cur)
            roc_auc_scores.append(roc_auc_cur)
            print(f'Metrics on {i_par} iteration on {i_fold} fold: LogLoss: {logloss_cur}, roc auc: {roc_auc_cur}')
            i_fold += 1
        mean_logloss = np.mean(logloss_scores)
        mean_roc_auc = np.mean(roc_auc_scores)
        print(f'Mean metrics on {i_par} iteration: LogLoss: {mean_logloss}, roc auc: {mean_roc_auc}')
        results.append({
            "params": params,
            "logloss": mean_logloss,
            "roc_auc": mean_roc_auc
        })
        i_par += 1

    # Сортировка по logloss (чем меньше, тем лучше)
    results = sorted(results, key=lambda x: x["logloss"])
    return results
