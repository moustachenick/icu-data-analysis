import pandas as pd
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler


class XGBoost_classifier: 

    """
    A class that uses the XGboost algorithm to predict abnormal Intracranial Pressure Values 
    """

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

    def normalization(self, exclude_columns=["icp_next","patient_id", "date_of_birth", "timestamp"]):
       
         """
         Normalize the data using Min-Max scaling while excluding specified columns.
         """
        
         columns_to_scale = [col for col in self.data.columns if col not in exclude_columns]
         scaler = MinMaxScaler()
         self.data[columns_to_scale] = scaler.fit_transform(self.data[columns_to_scale])

         print(" Data successfully normalized.")
         return self.data
    

    
