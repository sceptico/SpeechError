'''
The DataLoader class is used to load the data from the data_path and return the data.

The DataLoader class has the following methods:
    - __init__(data_path): Constructor that initializes the data_path attribute.
    - load_data(): Method that loads the data from the data_path.
    - get_data(): Method that returns the loaded data.
'''

# Local imports
from .logger_setup import logger

# Third party imports
import pandas as pd

class DataLoader:
    def __init__(self, data_path):
        self.data_path = data_path
        self.data = self.load_data()

    def load_data(self):
        try:
            data = pd.read_csv(self.data_path)
        except Exception as e:
            logger.error(f"Error loading data from {self.data_path}: {e}")
            data = None
        return data

    def get_data(self):
        return self.data