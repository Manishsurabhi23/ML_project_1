import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models


@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join("artifacts", "model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting training and testing data")

            X_train, y_train = train_array[:, :-1], train_array[:, -1]
            X_test, y_test = test_array[:, :-1], test_array[:, -1]

            models = {
                "Random Forest": RandomForestRegressor(random_state=42),
                "Decision Tree": DecisionTreeRegressor(random_state=42),
                "Gradient Boosting": GradientBoostingRegressor(random_state=42),
                "Linear Regression": LinearRegression(),
                "XGB Regressor": XGBRegressor(
                    objective="reg:squarederror",
                    eval_metric="rmse",
                    random_state=42
                ),
                "CatBoost Regressor": CatBoostRegressor(
                    verbose=False,
                    random_state=42
                ),
                "AdaBoost Regressor": AdaBoostRegressor(random_state=42),
                "KNeighbors Regressor": KNeighborsRegressor(),
                "Lasso": Lasso(),
                "Ridge": Ridge(),
                "ElasticNet": ElasticNet()
            }

            params = {
                "Random Forest": {
                    "n_estimators": [100, 200, 300],  # Added 300
                    "max_depth": [None, 10, 20, 30],  # Added 30
                    "min_samples_split": [2, 5, 10],  # Added 10
                    "min_samples_leaf": [1, 2, 4],  # Added 4
                    "max_features": ["sqrt", "log2", None]  # Added None
                },

                "Decision Tree": {
                    "max_depth": [None, 5, 10, 20, 30],  # Added more options
                    "min_samples_split": [2, 5, 10],
                    "min_samples_leaf": [1, 2, 4],
                    "criterion": ["squared_error", "friedman_mse"]  # Added criterion
                },

                "Gradient Boosting": {
                    "n_estimators": [100, 200, 300],  # Added 300
                    "learning_rate": [0.01, 0.05, 0.1],  # Added 0.01
                    "max_depth": [3, 5, 7],  # Added 7
                    "subsample": [0.8, 1.0],  # Added subsample
                    "min_samples_split": [2, 5]  # Added
                },

                "Linear Regression": {},  # No hyperparameters to tune

                "XGB Regressor": {
                    "n_estimators": [100, 200, 300],
                    "learning_rate": [0.01, 0.05, 0.1],  # Added 0.01
                    "max_depth": [3, 5, 7],  # Added 7
                    "subsample": [0.7, 0.8, 1.0],  # Added 0.7
                    "colsample_bytree": [0.7, 0.8, 1.0],  # Added 0.7
                    "min_child_weight": [1, 3, 5],  # Added
                    "gamma": [0, 0.1, 0.2]  # Added
                },

                "CatBoost Regressor": {
                    "iterations": [100, 200, 300],  # Reduced max from 500
                    "learning_rate": [0.01, 0.05, 0.1],  # Added 0.01
                    "depth": [4, 6, 8],  # Added 8
                    "l2_leaf_reg": [1, 3, 5]  # Added regularization
                },

                "AdaBoost Regressor": {
                    "n_estimators": [50, 100, 200],  # Added 200
                    "learning_rate": [0.01, 0.05, 0.1, 1.0]  # Added more options
                },

                "KNeighbors Regressor": {
                    "n_neighbors": [3, 5, 7, 9, 11],  # Added 9, 11
                    "weights": ["uniform", "distance"],
                    "algorithm": ["auto", "ball_tree", "kd_tree"]  # Added
                },

                "Lasso": {
                    "alpha": [0.0001, 0.001, 0.01, 0.1, 1, 10],  # Added more
                    "max_iter": [1000, 2000, 5000],
                    "selection": ["cyclic", "random"]  # Added
                },

                "Ridge": {
                    "alpha": [0.01, 0.1, 1, 10, 100],  # Added more options
                    "solver": ["auto", "svd", "cholesky"]  # Added
                },

                "ElasticNet": {
                    "alpha": [0.0001, 0.001, 0.01, 0.1, 1],  # Added 0.0001
                    "l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9],  # Added more
                    "max_iter": [1000, 2000, 5000],
                    "selection": ["cyclic", "random"]  # Added
                }
            }

            model_report, trained_models = evaluate_models(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                models=models,
                params=params
            )

            best_model_name = max(model_report, key=model_report.get)
            best_model_score = model_report[best_model_name]
            best_model = trained_models[best_model_name]

            logging.info(
                f"Best model found: {best_model_name} with R2 Score: {best_model_score}"
            )

            if best_model_score < 0.6:
                raise CustomException(
                    "No model achieved R2 score greater than 0.6",
                    sys
                )

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted = best_model.predict(X_test)
            r2 = r2_score(y_test, predicted)

            return r2

        except Exception as e:
            logging.error("Exception occurred in ModelTrainer")
            raise CustomException(e, sys)
