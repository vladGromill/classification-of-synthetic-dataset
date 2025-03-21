import re
import os
import sys
import json
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss

PROJECT_DIR = Path(os.getcwd())
sys.path.append(str(Path(PROJECT_DIR, 'utils')))
# import make_full_pipeline
# import importlib
# importlib.reload(make_full_pipeline)
from make_full_pipeline import Pipeline_log_regression, PipelineCatBoostClassifier
from hyperparameters_tuning import custom_grid_search_catboost
X_train = pd.read_csv(str(Path(PROJECT_DIR, 'ds_problem', 'problem_train.csv')), low_memory=False)
labels = pd.read_csv(str(Path(PROJECT_DIR, 'ds_problem', 'problem_labels.csv')))
X_test = pd.read_csv(str(Path(PROJECT_DIR, 'ds_problem', 'problem_test.csv')), low_memory=False)


param_distributions = {
    'use_feature_selection': [True, False],  # Использовать ли выбор признаков
    'min_filled_ratio': [0.6],  # Минимальная заполненность для отбора признаков
    'model_params__iterations': [100],
    'model_params__depth': [6, 8],
    'model_params__learning_rate': [0.05, 0.1],
    'model_params__l2_leaf_reg': [3, 5],
    'model_params__border_count': [32, 64],
    'calibration_method': ['isotonic'],  # Способ калибровки
    'cv': [2],  # Количество фолдов для калибровки
}
best_params_for_all_labels = {i:{} for i in range(1, labels.shape[1])}
all_params = []
for i in range(2, labels.shape[1]): # начинаю со второго таргета, т. к. для первого отдельно уже посчитал
    y = labels.iloc[:, i].values
    
    pipeline = PipelineCatBoostClassifier()
    results = custom_grid_search_catboost(X_train, y, pipeline, param_distributions, cv_splits=3)

    # Лучшие параметры
    best_result = results[0]
    best_params_for_all_labels[i] = best_result
    all_params.append(results)
    
    print(f"Лучшие параметры для {i}-го labels:", best_result["params"])
    print(f"Лучший LogLoss для {i}-го labels:", best_result["logloss"])
    print(f"Лучший ROC AUC для {i}-го labels:", best_result["roc_auc"])


# Определяем пути для сохранения файлов
best_params_file = "best_params.json"
all_params_file = "all_params.json"

# Сохраняем best_params_for_all_labels
with open(best_params_file, "w") as f:
    json.dump(best_params_for_all_labels, f, indent=4)

# Сохраняем all_params
with open(all_params_file, "w") as f:
    json.dump(all_params, f, indent=4, default=str)