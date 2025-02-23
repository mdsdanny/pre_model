"""
In supervised learning, Gradient Boosting provides a regression/classification
technique for aggregating the outcome of multiple decision trees. Is a sequential method
that aims to improve the performance of each subsequent tree. (not in parallel)
"""
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from core.model import Model
from scrubbing import DataScrubbing
from validations import SplitValidation

class GradientBoosting(Model):

    def init_model(self):
        self.model = GradientBoostingClassifier(
            n_estimators= 250,
            learning_rate = 0.1,
            max_depth = 5,
            min_samples_split = 4,
            min_samples_leaf = 6,
            max_features = 0.6,
            loss = 'exponential'
        )

    def reload_model(self, n_estimators = 350, loss='huber'):
        self.model = GradientBoostingClassifier(
            n_estimators= n_estimators,
            learning_rate = 0.1,
            max_depth = 5,
            min_samples_split = 4,
            min_samples_leaf = 6,
            max_features = 0.6,
            loss = loss
        )


