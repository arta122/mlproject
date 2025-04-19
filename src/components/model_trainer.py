import os
import sys
from dataclasses import dataclass
from catboost import CatBoostRegressor
from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from src.exception import CustomException
from src.logger import logging
from src.utils import save_obj, evaluate_model

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts', "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_arr, test_arr, preprocessor_path):
        try:
            logging.info('Splitting training and testing data')
            x_train, y_train, x_test, y_test = (
                train_arr[:, :-1],
                train_arr[:, -1],
                test_arr[:, :-1],
                test_arr[:, -1]
            )

            models = {
                'knn': KNeighborsRegressor(),
                'gradient': GradientBoostingRegressor(),
                'linear regression': LinearRegression(),
                'random forest': RandomForestRegressor(),
                'ada boost regressor': AdaBoostRegressor(),
                'decision tree': DecisionTreeRegressor(),
                'xgboost regression': XGBRegressor(),
                'catboost regressor': CatBoostRegressor(verbose=False)
            }
            params = {
            'knn': {'n_neighbors': [3, 5, 7]},
            'gradient': {'n_estimators': [100, 200], 'learning_rate': [0.1, 0.05]},
            'linear regression': {},
            'random forest': {'n_estimators': [100, 200], 'max_depth': [None, 10, 20]},
            'ada boost regressor': {'n_estimators': [50, 100], 'learning_rate': [0.01, 0.1, 1]},
            'decision tree': {'max_depth': [None, 10, 20]},
            'xgboost regression': {'n_estimators': [100, 200], 'learning_rate': [0.05, 0.1]},
            'catboost regressor': {'depth': [6, 10], 'learning_rate': [0.03, 0.1], 'iterations': [100, 200]}
        }

            model_report: dict = evaluate_model(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test, models=models,param =params)
            best_model_score = max(model_report.values())

            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No best model found")

            logging.info("Best model found on both training and testing set")

            save_obj(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predictions = best_model.predict(x_test)
            final_r2_score = r2_score(y_test, predictions)
            return final_r2_score

        except Exception as e:
            raise CustomException(e, sys)
