import pandas as pd
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler


class XGBoost_classifier: 

    """
    A class that uses the XGboost algorithm to predict abnormal Intracranial Pressure Values 
    """

    import pandas as pd
from pathlib import Path
import os
from sklearn.preprocessing import MinMaxScaler

def __init__(self, data=None, file_path=None):
       
        self.data = data
        self.file_path = file_path

        if self.data is None and self.file_path is not None:
            self.load_data()

def load_data(self):
    
        try:
            self.data = pd.read_csv(self.file_path)
            print(f"Data successfully loaded from {self.file_path}")
        except Exception as e:
            print(f"Error loading data: {e}")

def normalization(self, data=None):
        """
        Normalize the data using Min-Max scaling.
        
        """
        scaler = MinMaxScaler()
        normalized_data = scaler.fit_transform(data)
        self.data = pd.DataFrame(normalized_data, columns=data.columns)
        print("Data successfully normalized.")
        return self.data