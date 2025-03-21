import logging
from typing import Type
import copy

# Data processing
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from numpy import ndarray
from sklearn.base import OneToOneFeatureMixin
from category_encoders import TargetEncoder

# Pipeline
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
# Regressors
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction import FeatureHasher
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.calibration import CalibratedClassifierCV

from sklearn.base import clone
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import log_loss, roc_auc_score


class FeatureSelection(BaseEstimator, TransformerMixin):
    def __init__(self, min_filled_ratio=0.8, cat_features_int=None):
        self.min_filled_ratio = min_filled_ratio
        self.cat_features_int = cat_features_int or ['n_0047', 'n_0050', 'n_0052', 'n_0061', 'n_0075', 'n_0091']
        self.filtered_features = []

    def fit(self, X, y=None):
        X = X.drop(columns=self.cat_features_int)
        if 'id' in X.columns:
            X = X.drop(columns=['id'])
        self.filtered_features = X.columns[X.notna().mean() >= self.min_filled_ratio].tolist()
        return self

    def transform(self, X):
        return X[self.filtered_features]


class MyTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, strategy_cat='most_frequent', strategy_num='mean', scaler=None, scaler2=None, use_pca=False, pca_percentage=0.9):
        self.strategy_cat = strategy_cat
        self.strategy_num = strategy_num
        self.num_imputer = None
        self.cat_inputer = None
        self.scaler = scaler or StandardScaler()
        self.scaler2 = scaler2 or StandardScaler()
        self.features_order = None
        self.use_pca = use_pca
        self.pca_percentage = pca_percentage
        self.encoder = TargetEncoder() # используем его, так как у нас будет обучаться 14 моделей для каждого из таргетов,
                                       # а TargetEncoder будет учитывать изменения в таргете и подстраивать категориальные признаки под него
        self.NUM_FEATURES_INT = []
        self.CAT_FEATURES_OBJECT = []
        self.NUM_FEATURES_FLOAT = []
    
    def fit(self, X, y):
        if 'o_0176' in X.columns:
            self.NUM_FEATURES_INT = ['o_0176']
        self.CAT_FEATURES_OBJECT = [column for column in X.columns if X[column].dtype == 'object']
        if 'o_0264' in X.columns:
            self.CAT_FEATURES_OBJECT.append('o_0264')
        self.NUM_FEATURES_FLOAT = [column for column in X.columns if X[column].dtype == 'float64']

        self.cat_features = self.CAT_FEATURES_OBJECT
        self.num_features = self.NUM_FEATURES_INT + self.NUM_FEATURES_FLOAT
        
        self.num_imputer = SimpleImputer(strategy=self.strategy_num)
        self.cat_inputer = SimpleImputer(strategy=self.strategy_cat)

        self.num_imputer.fit(X[self.num_features])
        self.cat_inputer.fit(X[self.cat_features])

        self.features_o = [col for col in self.num_features if col[0] == 'o'] # только для этих признаков нужна нормализация
        #features_n = [col for col in num_filtered_features if col[0] == 'n'] # значения этих признаков находятся в в диапазоне [0, 1]

        self.encoder.fit(X[self.cat_features], y)
        X_imputed = pd.DataFrame(
            self.num_imputer.transform(X[self.num_features]), 
            columns=self.num_features,
            index=X.index
        )

        if self.scaler:
            self.scaler.fit(X_imputed[self.features_o])

        #X_all_num = pd.concat(
        #    [X_imputed_o, X[self.num_features].drop(columns=self.features_o, errors='ignore')],
        #axis=1
        #)
        
        if self.use_pca:
            self.pca = PCA()  # Создаем объект PCA
            explained_variance = self.pca.fit(X_imputed).explained_variance_ratio_.cumsum()
            n_components = (explained_variance >= self.pca_percentage).argmax() + 1
            self.pca = PCA(n_components=n_components)
            self.pca.fit(X_imputed)
        
        self.features_order = self.num_features + self.cat_features

        return self

    def transform(self, X):
        missing_features = set(self.num_features + self.cat_features) - set(X.columns)
        if missing_features:
            raise ValueError(f"Missing features during transform: {missing_features}")
        
        X = X[self.features_order]

        X_num = pd.DataFrame(
            self.num_imputer.transform(X[self.num_features]),
            columns=self.num_features,
            index=X.index
        )

        X_cat = pd.DataFrame(
            self.cat_inputer.transform(X[self.cat_features]),
            columns=self.cat_features,
            index=X.index
        )

        X_cat = self.encoder.transform(X_cat)

        X_scaled_o = self.scaler.transform(X_num[self.features_o]) # нормализация только для features_o
        X_scaled_o = pd.DataFrame(X_scaled_o, columns=self.features_o, index=X.index)
        #X_scaled = pd.concat(
        #    [X_scaled_o, X_num.drop(columns=self.features_o, errors='ignore')],
        #    axis=1)
        X_num.update(X_scaled_o)

        X_cat_scaled = self.scaler2.fit_transform(X_cat)  # Нормализация
        X_cat = pd.DataFrame(X_cat_scaled, columns=self.cat_features, index=X.index)

        if self.use_pca:
            X_num = self.pca.transform(X_num)
            X_num = pd.DataFrame(X_num, columns=[f'PC{i+1}' for i in range(X_num.shape[1])], index=X.index)

            scaler_pca = StandardScaler()
            X_num = pd.DataFrame(
                scaler_pca.fit_transform(X_num),
                columns=X_num.columns,
                index=X_num.index)

        
        return pd.concat([X_num, X_cat], axis=1)



class Pipeline_log_regression(BaseEstimator):
    def __init__(self, feature_selection_params=None, my_transformer_params=None, logistic_regression_params=None):
        self.feature_selection_params = feature_selection_params or {}
        self.my_transformer_params =  my_transformer_params or {}
        self.logistic_regression_params = logistic_regression_params or {}
        self.pipeline = None

    def fit(self, X, y):
        feature_selector = FeatureSelection(**self.feature_selection_params)
        transformer = MyTransformer(**self.my_transformer_params)
        base_model = LogisticRegression(**self.logistic_regression_params)
        #print(self.feature_selection_params)
        #print(self.my_transformer_params)
        #print(self.logistic_regression_params)
        # Создание пайплайна
        self.pipeline = Pipeline([
            ('feature_selector', feature_selector),
            ('transformer', transformer),
            ('base_model', base_model)
        ])

        self.pipeline.fit(X, y)
        return self

    def predict(self, X):
        return self.pipeline.predict(X)

    def predict_proba(self, X):
        return self.pipeline.predict_proba(X)

    def get_params(self, deep=True):
        return {
            'feature_selection_params': self.feature_selection_params,
            'my_transformer_params': self.my_transformer_params,
            'logistic_regression_params': self.logistic_regression_params
        }

    def set_params(self, **params):
        feature_selection_params = {}
        my_transformer_params = {}
        logistic_regression_params = {}

        # Распределяем параметры по компонентам
        for key, value in params.items():
            if key.startswith('feature_selection_params__'):
                param_name = key.split('__', 1)[1]  # Извлекаем реальное имя параметра
                feature_selection_params[param_name] = value
            elif key.startswith('my_transformer_params__'):
                param_name = key.split('__', 1)[1]
                my_transformer_params[param_name] = value
            elif key.startswith('logistic_regression_params__'):
                param_name = key.split('__', 1)[1]
                logistic_regression_params[param_name] = value

        # Устанавливаем параметры
        self.feature_selection_params = feature_selection_params
        self.my_transformer_params = my_transformer_params
        self.logistic_regression_params = logistic_regression_params

        self.pipeline = Pipeline([
            ('feature_selector', FeatureSelection(**self.feature_selection_params)),
            ('transformer', MyTransformer(**self.my_transformer_params)),
            ('base_model', LogisticRegression(**self.logistic_regression_params))
        ])
        
    def evaluate_metrics(self, X, y_true):

        y_pred_proba = self.pipeline.predict_proba(X)
        log_loss_score = log_loss(y_true, y_pred_proba)

        y_pred = self.pipeline.predict(X)
        roc_auc = roc_auc_score(y_true, y_pred)

        return {
            'logloss': log_loss_score,
            'roc_auc': roc_auc
        }


class PipelineCatBoostClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, use_feature_selection=False, min_filled_ratio=0.8, model_params=None, calibration_method='isotonic', cv=2):
        self.use_feature_selection = use_feature_selection
        self.min_filled_ratio = min_filled_ratio
        self.model_params = model_params or {
            'iterations': 100, 'depth': 8, 'learning_rate': 0.1, 'l2_leaf_reg': 3, 'border_count': 64,
            'loss_function': 'Logloss', 'eval_metric': 'Logloss', 'verbose': 0
        }
        self.calibration_method = calibration_method
        self.cv = cv
        self.feature_selector = None
        self.model = None
        self.calibrated_model = None

    def set_params(self, **params):
        """Устанавливает параметры для RandomizedSearchCV."""
        for param, value in params.items():
            if param.startswith("model_params__"):  # Параметры CatBoost
                key = param.split("__", 1)[1]
                self.model_params[key] = value
            else:
                setattr(self, param, value)
        return self

    def get_params(self, deep=True):
        """Возвращает параметры для интеграции с RandomizedSearchCV."""
        params = {
            'use_feature_selection': self.use_feature_selection,
            'min_filled_ratio': self.min_filled_ratio,
            'model_params': self.model_params,
            'calibration_method': self.calibration_method,
            'cv': self.cv,
        }
        if deep:
            params.update({f"model_params__{k}": v for k, v in self.model_params.items()})
        return params

    def fit(self, X, y):
        if self.use_feature_selection:
            self.feature_selector = FeatureSelection(min_filled_ratio=self.min_filled_ratio)
            X = self.feature_selector.fit_transform(X)
        
        self.filtered_columns = X.columns
        self.cat_features = [column for column in X.columns if X[column].dtype == 'object']
        
        for col in self.cat_features:
            X.loc[:, col] = X[col].fillna('missing')
        
        self.model = CatBoostClassifier(cat_features=self.cat_features, **self.model_params)
        self.model.fit(X, y)

        self.calibrated_model = CalibratedClassifierCV(self.model, method=self.calibration_method, cv=self.cv)
        self.calibrated_model.fit(X, y)
        return self

    def predict(self, X):
        if self.use_feature_selection:
            X = X[self.filtered_columns]

        self.cat_features = [column for column in X.columns if X[column].dtype == 'object']
        for col in self.cat_features:
            X.loc[:, col] = X[col].fillna('missing')
        return self.calibrated_model.predict(X)

    def predict_proba(self, X, calibrate=True):
        if self.use_feature_selection:
            X = X[self.filtered_columns]

        self.cat_features = [column for column in X.columns if X[column].dtype == 'object']
        
        for col in self.cat_features:
            X.loc[:, col] = X[col].fillna('missing').astype(str)
        
        if calibrate:
            return self.calibrated_model.predict_proba(X)
        else:
            return self.model.predict_proba(X)

    def evaluate_metrics(self, X, y_true, calibrate=True):
        if self.use_feature_selection:
            X = X[self.filtered_columns]
        for col in self.cat_features:
            X.loc[:, col] = X[col].fillna('missing')
        
        if calibrate:
            y_pred_proba = self.calibrated_model.predict_proba(X)[:, 1]
            log_loss_score = log_loss(y_true, y_pred_proba)
            y_pred = self.calibrated_model.predict(X)
            roc_auc = roc_auc_score(y_true, y_pred)
        else:
            y_pred_proba = self.model.predict_proba(X)[:, 1]
            log_loss_score = log_loss(y_true, y_pred_proba)
            y_pred = self.model.predict(X)
            roc_auc = roc_auc_score(y_true, y_pred)
        
        return {
            'logloss': log_loss_score,
            'roc_auc': roc_auc
        }


    
'''
class PipelineCatBoostClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, use_feature_selection=False, min_filled_ratio=0.8, model_params=None, calibration_method='sigmoid', cv=2):
        self.use_feature_selection = use_feature_selection
        self.min_filled_ratio = min_filled_ratio
        self.model_params = model_params or {
            'iterations': 100, 'depth': 6, 'learning_rate': 0.1, 'loss_function': 'Logloss', 'eval_metric': 'Logloss'
        }
        self.calibration_method = calibration_method
        self.cv = cv
        self.pipeline_without_calibration = None
        self.pipeline_with_calibration = None

    def _build_pipeline(self, X):
        # Определение категориальных признаков
        cat_features = X.select_dtypes(include=['object']).columns.tolist()

        # Feature selector
        def feature_selector(X):
            if not self.use_feature_selection:
                return X
            filled_ratios = X.notnull().mean(axis=0)
            selected_features = filled_ratios[filled_ratios >= self.min_filled_ratio].index
            return X[selected_features]

        feature_selection_step = FunctionTransformer(feature_selector)

        # Preprocessing for categorical features only
        cat_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing'))
        ])

        preprocessor = ColumnTransformer(
            transformers=[
                ('cat', cat_transformer, cat_features)  # Обработка категориальных признаков
            ],
            remainder='passthrough'  # Остальные признаки передаются без изменений
        )

        # CatBoost model
        catboost_model = CatBoostClassifier(cat_features=cat_features, **self.model_params)

        # Assemble pipeline
        pipeline = Pipeline(steps=[
            ('feature_selection', feature_selection_step),
            ('preprocessing', preprocessor),
            ('catboost', catboost_model)
        ])
        return pipeline


    def fit(self, X, y):
        # Build and fit the uncalibrated pipeline
        self.pipeline_without_calibration = self._build_pipeline(X)
        self.pipeline_without_calibration.fit(X, y)

        # Add calibration to the pipeline
        self.pipeline_with_calibration = Pipeline(steps=[
            ('uncalibrated', self.pipeline_without_calibration),
            ('calibration', CalibratedClassifierCV(self.pipeline_without_calibration['catboost'], method=self.calibration_method, cv=self.cv))
        ])
        self.pipeline_with_calibration.fit(X, y)
        return self

    def predict(self, X):
        return self.pipeline_with_calibration.predict(X)

    def predict_proba(self, X, calibrated=True):
        pipeline = self.pipeline_with_calibration if calibrated else self.pipeline_without_calibration
        return pipeline.predict_proba(X)

'''