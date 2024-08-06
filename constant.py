import platform
import os
import sys
from pathlib import Path

if getattr(sys, 'frozen', False):
    application_path = os.path.dirname(sys.executable)
elif __file__:
    application_path = os.path.dirname(__file__)
else:
    raise Exception("ERROR")
BASE_PATH = Path(os.path.abspath(application_path))

PATH_SEPARATOR = "\\" if platform.system() == "Windows" else "/"

DEVICE = "cuda:0"

LOGGING_DIR = BASE_PATH / "logs"
LOGGING_LEVEL = "DEBUG"
