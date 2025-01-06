"""
K-Nearest Neighbors is a classification technique, which classifies new unknown data
points based on their proximity to know datapoints. Is determined by setting "k" number
of data points closest to the target datapoint.
"""
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from core.model import Model

class KNearestNeighbors(Model):

    def new_model(self, n):
        """
        Creates and assigns a new KNeighborsClassifier.
        :return: None
        """
        self.model = KNeighborsClassifier(n_neighbors=n)

    def scale(self):
        """
        #TODO description
        :return: Scaled data
        """
        scaler = StandardScaler()
        scaler.fit(self.df.data_frame().drop('Clicked on Ad', axis=1))
        return scaler.transform(self.df.data_frame().drop('Clicked on Ad', axis=1))






