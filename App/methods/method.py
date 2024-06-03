
import pandas as pd
import numpy as np


class BaseMethodMeta(type):
    """
    Metaclass register implemented methods automaticly. This allows the usage of runtime arguments to specify the method to run experiments.

    Attributes:
        registry (dict): Registry to store the registered methods.

    """
    registry = {}

    def __init__(cls, name, bases, attrs):
        super().__init__(name, bases, attrs)
        if name != 'BaseMethod':
            BaseMethodMeta.registry[name] = cls