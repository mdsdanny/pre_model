"""
In supervised learning, Random forests involves growing multiple decision trees
using a randomized selection of input data for each tree and combining the results
by averaging the output for regression or class voting for classification.
"""
from sklearn.ensemble import RandomForestClassifier
from core.model import Model

class RandomForests(Model):

    def init_model(self):
        """
        Creates and assigns a new Decision Tree Classifier.
        :return: None
        """
        self.model = RandomForestClassifier()

