"""
K-means clustering (k-Means) is a dimension reduction technique.
Identifies groups of data points without knowledge of existing classes.
"""
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from core import PreModel

class Kmeans(PreModel):

    def __init__(self, n_samples, n_features, centers):
        """
        Initializes an artificial dataset variables.
        :param n_samples: Number of samples
        :param n_features: Number of features
        :param centers: Number of centers
        """
        super().__init__()
        self.n_samples = n_samples
        self.n_features = n_features
        self.centers = centers
        self.model_predict = None
        self.centroids = None
        self.X = None
        self.y = None

    def k_means(self, n_clusters, cluster_std, random_state):
        """
        Generates an artificial dataset with all variables values specified in initialization.
        Discovers natural groupings among datapoints that shares similar attributes.
        :param n_clusters: Number of clusters to use
        :param cluster_std: Number of cluster standard deviation
        :param random_state: Random number generation
        :return: None
        """
        self.X, self.y = make_blobs(n_samples=self.n_samples, n_features=self.n_features, centers=self.centers,
                               cluster_std=cluster_std, random_state=random_state)
        self.pre_model = KMeans(n_clusters=n_clusters)
        self.pre_model.fit(self.X)

    def predict(self):
        """
        Generates the centroid coordinates (cluster centers).
        :return: None
        """
        #The KMeans model
        self.model_predict = self.pre_model.predict(self.X)
        self.centroids = self.pre_model.cluster_centers_

    def visualize(self, size, s, cmap_s, centroids_size):
        """
        Displays the clusters on a scatterplot using two sets of elements (model_predict and centroids).
        :param size: A tuple of values for a figure
        :param s: scalar marker size
        :param cmap_s: Color Map
        :param centroids_size: centroids marker size
        :return: A pyplot instance
        """
        plt.figure(figsize=size)
        plt.scatter(self.X[:, 0], self.X[:, 1], c=self.model_predict, s=s, cmap=cmap_s)
        plt.scatter(self.centroids[:, 0], self.centroids[:, 1], c='black', s=centroids_size, alpha=1)
        return plt

    def render(self, mpyplot, x_label, y_label, size=None):
        """
        It renders the matplotlib.pyplot into distortions visualization.
        :param mpyplot: A matplotlib.pyplot instance
        :param x_label: Label of X axis
        :param y_label: Label of Y axis
        :param size: A tuple of values for a figure
        :return: None
        """
        distortions = []
        r = range(1, 10)
        for k in r:
            model = KMeans(n_clusters=k)
            model.fit(self.X, self.y)
            distortions.append(model.inertia_)

        mpyplot.figure(figsize=size)
        mpyplot.plot(r, distortions)
        mpyplot.xlabel(x_label)
        mpyplot.ylabel(y_label)
        mpyplot.show()

