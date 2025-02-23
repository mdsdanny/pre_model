"""
In supervised learning, SVM is a regression analysis technique.
Analyses  complex data and downplaying the influence of outliers.
"""
from sklearn.svm import SVC
from core.model import Model
from grid_search import GridSearch
from scrubbing import DataScrubbing
from validations import SplitValidation

class SupportVectorMachines(Model):

    def init_model(self):
        """
        Creates and assigns a new Support Vector Classifier.
        :return: None
        """
        self.model = SVC()

