from abc import ABC, abstractmethod


class Regression(ABC):

    @abstractmethod
    def fit(self, X_train, y_train):
        pass

    @abstractmethod
    def predict(self, X_test):
        pass
