from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.model_selection import cross_val_score
import numpy as np

import xgboost as xgb

class ClassificationPredictor:

    """
    A class that uses the XGboost algorithm to predict abnormal Intracranial Pressure Values
    """

    def run_pipeline(self, X_train, X_test, y_train, y_test):
        """
        Run the pipeline to train and evaluate the model.
        Args:
            X_train: Training features.
            X_test: Testing features.
            y_train: Training target variable.
            y_test: Testing target variable.
        """

        # Normalize the data before training
        X_train = self.normalization(X_train)
        X_test = self.normalization(X_test)


        print(X_train.head())

        # Continue with training and evaluation
        model = self.classification(X_train, X_test, y_train, y_test)
        return model

    def normalization(self, data, exclude_columns=None):
        """
        Normalize the data using Min-Max scaling while excluding specified columns.

        Args:
            data (pd.DataFrame): The data to be normalized.
            exclude_columns (list, optional): List of columns to exclude from normalization.

        Returns:
            pd.DataFrame: The normalized data.
        """
        if exclude_columns is None:
            exclude_columns = []
        exclude_columns.extend(["icp_binary", "patient_id", "date_of_birth", "timestamp"]) #Exclude the target ICP column from normalization as it has binary values and we don't want them normalized  
        
        
        columns_to_scale = [col for col in data.columns if col not in exclude_columns]
        scaler = MinMaxScaler()
        data[columns_to_scale] = scaler.fit_transform(data[columns_to_scale])

        print("Data successfully normalized.")
        return data
    
    def classification(self, X_train, X_test, y_train, y_test):

        # Train XGBoost model
        model = xgb.XGBClassifier()
        model.fit(X_train, y_train)

        # Make predictions
        y_pred = model.predict(X_test)

        print("Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred))

        print("Classification Report:")
        print(classification_report(y_test, y_pred))

         # 10-Fold Cross-Validation on Training Set 
        print("Performing 10-fold Cross Validation")

        cv_scores = cross_val_score(model, X_train, y_train, cv=10, scoring="accuracy")

        print("CV Scores:", cv_scores)
        print("CV Mean Accuracy:", np.mean(cv_scores))

        return model
        