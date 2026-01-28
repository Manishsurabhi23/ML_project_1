import os
import sys
import dill
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import r2_score

from src.exception import CustomException
from src.logger import logging


def save_object(file_path: str, obj: object) -> None:
    """
    Save any Python object using dill
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

        logging.info(f"Object saved successfully at {file_path}")

    except Exception as e:
        logging.error("Exception occurred while saving object")
        raise CustomException(e, sys)
    
def load_object(file_path: str) -> object:
    """
    Load a Python object saved using dill
    """
    try:
        with open(file_path, "rb") as file_obj:
            obj = dill.load(file_obj)

        logging.info(f"Object loaded successfully from {file_path}")
        return obj

    except Exception as e:
        logging.error("Exception occurred while loading object")
        raise CustomException(e, sys)


def evaluate_models(
    X_train,
    y_train,
    X_test,
    y_test,
    models: dict,
    params: dict
):
    """
    Train models using GridSearchCV with K-Fold
    Returns:
        model_report: dict (model_name -> r2_score)
        trained_models: dict (model_name -> trained best model)
    """
    try:
        model_report = {}
        trained_models = {}

        kf = KFold(
            n_splits=5,
            shuffle=True,
            random_state=42
        )

        for model_name, model in models.items():
            logging.info(f"Training model: {model_name}")

            param_grid = params.get(model_name)

            if param_grid:
                gs = GridSearchCV(
                    estimator=model,
                    param_grid=param_grid,
                    scoring="r2",
                    cv=kf,
                    n_jobs=-1
                )
                gs.fit(X_train, y_train)

                best_model = gs.best_estimator_
                logging.info(f"{model_name} best params: {gs.best_params_}")
            else:
                best_model = model

            # 🔒 Always fit before predict
            best_model.fit(X_train, y_train)

            y_pred = best_model.predict(X_test)
            r2 = r2_score(y_test, y_pred)

            model_report[model_name] = r2
            trained_models[model_name] = best_model

            logging.info(f"{model_name} R2 Score: {r2}")

        return model_report, trained_models

    except Exception as e:
        logging.error("Exception occurred while evaluating models")
        raise CustomException(e, sys)
