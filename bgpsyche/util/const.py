import os.path
from pathlib import Path
import configparser

HERE = Path(os.path.dirname(__file__)) / '..'
DATA_DIR = HERE / 'data'

ENV = configparser.ConfigParser()
ENV.read(HERE / 'env.ini')