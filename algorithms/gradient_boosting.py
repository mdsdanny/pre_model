"""
In supervised learning, Gradient Boosting TODO.
"""
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from core.model import Model

class GradientBoosting(Model):

    def new_model(self):
        """
        Creates and assigns a new Decision Tree Classifier.
        :return: None
        """
        self.model = GradientBoostingClassifier(
            n_estimators= 250,
            learning_rate = 0.1,
            max_depth = 5,
            min_samples_split = 4,
            min_samples_leaf = 6,
            max_features = 0.6,
            loss = 'exponential'
        )

    def new_model2(self):
        """
        Creates and assigns a new Decision Tree Classifier.
        :return: None
        """
        self.model = GradientBoostingRegressor(
            n_estimators=350,
            learning_rate=0.1,
            max_depth=5,
            min_samples_split=4,
            min_samples_leaf=6,
            max_features=0.6,
            loss='huber'
        )



