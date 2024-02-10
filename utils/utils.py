from sysconfig import get_python_version
from enum import auto, Enum

def is_notebook() -> bool:
    try:
        shell = get_python_version().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False      # Probably standard Python interpreter

class DatasetType(Enum):
    TRAIN = auto()
    VALIDATION = auto()