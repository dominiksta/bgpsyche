import os.path
from pathlib import Path
import configparser

import joblib

HERE = Path(os.path.dirname(__file__)) / '..'
DATA_DIR = HERE / 'data'

JOBLIB_MEMORY = joblib.Memory(location=DATA_DIR / 'cache' / 'joblib')

ENV = configparser.ConfigParser()
ENV.read(HERE / 'env.ini')