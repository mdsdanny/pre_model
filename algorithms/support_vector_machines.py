"""
In supervised learning, SVM is a regression analysis technique.
Analyses  complex data and downplaying the influence of outliers.
"""
from sklearn.svm import SVC
from core.model import Model

class SupportVectorMachines(Model):

    def new_model(self):
        """
        Creates and assigns a new Support Vector Classifier.
        :return: None
        """
        self.model = SVC()

