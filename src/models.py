import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score
import warnings
warnings.filterwarnings('ignore')


class ModelFactory:

    @staticmethod
    def create_model(model_type: str, params: dict):
        if model_type == "xgboost":
            from xgboost import XGBClassifier
            return XGBClassifier(**params)

        elif model_type == "lightgbm":
            from lightgbm import LGBMClassifier
            return LGBMClassifier(**params)

        elif model_type == "catboost":
            from catboost import CatBoostClassifier
            return CatBoostClassifier(**params, verbose=0)

        else:
            raise ValueError(f"Unknown model type: {model_type}")


class BehaviorClassifier:

    def __init__(self, config):
        self.config = config
        self.models = {}
        self.thresholds = {}

    def train_action(self, X: pd.DataFrame, y: pd.Series, groups: pd.Series,
                    action: str) -> Dict:
        valid_mask = ~y.isna()
        X_valid = X[valid_mask]
        y_valid = y[valid_mask].astype(int)
        groups_valid = groups[valid_mask]

        if len(y_valid) == 0 or y_valid.sum() == 0:
            print(f"  {action}: No positive examples, skipping")
            return None

        cv = StratifiedGroupKFold(n_splits=self.config.N_SPLITS)
        oof_preds = np.zeros(len(y_valid))

        for fold, (train_idx, val_idx) in enumerate(cv.split(X_valid, y_valid, groups_valid)):
            X_train, X_val = X_valid.iloc[train_idx], X_valid.iloc[val_idx]
            y_train, y_val = y_valid.iloc[train_idx], y_valid.iloc[val_idx]

            model = ModelFactory.create_model(self.config.MODEL_TYPE, self.config.MODEL_PARAMS)
            model.fit(X_train, y_train)

            oof_preds[val_idx] = model.predict_proba(X_val)[:, 1]

        best_f1 = 0
        best_threshold = 0.5

        for threshold in np.arange(0.1, 0.9, 0.05):
            f1 = f1_score(y_valid, oof_preds >= threshold, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold

        final_model = ModelFactory.create_model(self.config.MODEL_TYPE, self.config.MODEL_PARAMS)
        final_model.fit(X_valid, y_valid)

        print(f"  {action}: F1={best_f1:.4f}, Threshold={best_threshold:.2f}, "
              f"Positives={y_valid.sum()}/{len(y_valid)}")

        return {
            'model': final_model,
            'threshold': best_threshold,
            'f1_score': best_f1,
            'oof_preds': oof_preds
        }

    def train_multiple_actions(self, X: pd.DataFrame, y: pd.DataFrame,
                               groups: pd.Series, action_list: List[str],
                               section_id: str) -> Dict[str, float]:
        f1_scores = {}

        for action in action_list:
            if action not in y.columns:
                continue

            result = self.train_action(X, y[action], groups, action)

            if result is not None:
                self.models[action] = result['model']
                self.thresholds[action] = result['threshold']
                f1_scores[action] = result['f1_score']

                self.save_model(action, section_id)

        return f1_scores

    def predict_action(self, X: pd.DataFrame, action: str) -> np.ndarray:
        if action not in self.models:
            return np.zeros(len(X))

        return self.models[action].predict_proba(X)[:, 1]

    def predict_multiple_actions(self, X: pd.DataFrame,
                                 action_list: List[str]) -> pd.DataFrame:
        predictions = pd.DataFrame(index=X.index)

        for action in action_list:
            predictions[action] = self.predict_action(X, action)

        return predictions

    def save_model(self, action: str, section_id: str):
        if action not in self.models:
            return

        path = self.config.get_model_path(section_id, action)
        joblib.dump({
            'model': self.models[action],
            'threshold': self.thresholds[action]
        }, path)

    def load_model(self, action: str, section_id: str):
        path = self.config.get_model_path(section_id, action)

        if not path.exists():
            print(f"Model not found: {path}")
            return False

        data = joblib.load(path)
        self.models[action] = data['model']
        self.thresholds[action] = data['threshold']
        return True

    def load_multiple_models(self, action_list: List[str], section_id: str):
        for action in action_list:
            self.load_model(action, section_id)


class EnsembleClassifier:

    def __init__(self, config):
        self.config = config
        self.classifiers = []

    def add_classifier(self, classifier: BehaviorClassifier):
        self.classifiers.append(classifier)

    def predict(self, X: pd.DataFrame, action_list: List[str],
               method: str = 'mean') -> pd.DataFrame:
        all_preds = []

        for classifier in self.classifiers:
            preds = classifier.predict_multiple_actions(X, action_list)
            all_preds.append(preds)

        if method == 'mean':
            return pd.concat(all_preds).groupby(level=0).mean()
        elif method == 'median':
            return pd.concat(all_preds).groupby(level=0).median()
        else:
            raise ValueError(f"Unknown ensemble method: {method}")
