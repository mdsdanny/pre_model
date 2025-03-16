"""
In supervised learning, Decision trees create a decision structure to interpret patterns
by splitting data into groups using variables that best split the data into homogenous
or numerically relevant groups based on entropy (a measure of variance in the data among
different classes).
"""
from sklearn.tree import DecisionTreeClassifier
from core.model import Model

class DecisionTree(Model):

    def init_model(self):
        """
       Creates and assigns a new Decision Tree Classifier.
       :return: None
       """
        self.model = DecisionTreeClassifier()




