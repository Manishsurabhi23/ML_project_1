import os
import sys
import pickle
import dill
from src.exception import CustomException
from src.logger import logging

import numpy as np
import pandas as pd

# Save a pickled object to a file
def save_object(file_path: str, obj: object) -> None:
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, 'wb') as file_obj:
            dill.dump(obj, file_obj)
        logging.info(f"Object saved successfully at {file_path}")
    except Exception as e:
        logging.info("Exception occurred while saving object")
        raise CustomException(e, sys)