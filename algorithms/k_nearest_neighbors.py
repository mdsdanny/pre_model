"""
K-Nearest Neighbors is a classification technique, which classifies new unknown data
points based on their proximity to know datapoints. Is determined by setting "k" number
of data points closest to the target datapoint.
"""
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from core.model import Model
from scrubbing import DataScrubbing
from validations import SplitValidation


class KNearestNeighbors(Model):

    def init_model(self):
        """
        Creates and assigns a new KNeighborsClassifier.
        :return: None
        """
        self.model = KNeighborsClassifier()

    def add_n_neighbors(self, n):
        """
        Creates and assigns a new KNeighborsClassifier.
        :return: None
        """
        self.model.n_neighbors = n

    def scale(self, independents):
        """
        #TODO description
        :return:
        :param independents:
        :return: Scaled data
        """
        scaler = StandardScaler()
        scaler.fit(independents)
        return scaler.transform(independents)
