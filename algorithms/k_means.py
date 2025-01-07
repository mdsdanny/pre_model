"""
K-means clustering (k-Means) is a dimension reduction technique.
Identifies groups of data points without knowledge of existing classes.
"""
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from core.model import Model

class Kmeans(Model):

    def __init__(self, n_samples, n_features, centers, cluster_std, random_state, csv_file=None):
        """
        Initializes an artificial dataset variables.
        :param n_samples: Number of samples
        :param n_features: Number of features
        :param centers: Number of centers
        :param cluster_std: Number of cluster standard deviation
        :param random_state: Random number generation
        """
        super().__init__(csv_file)
        self.n_samples = n_samples
        self.n_features = n_features
        self.centers = centers
        self.cluster_std = cluster_std
        self.random_state = random_state
        self.X = None
        self.y = None

    def k_means(self, n_clusters):
        """
        Generates an artificial dataset with all variables values specified in initialization.
        Discovers natural groupings among datapoints that shares similar attributes.
        :param n_clusters: Number of clusters to use
        :return: The KMeans model
        """
        self.X, self.y = make_blobs(n_samples=self.n_samples, n_features=self.n_features, centers=self.centers,
                               cluster_std=self.cluster_std, random_state=self.random_state)
        self.model = KMeans(n_clusters=n_clusters)
        self.model.fit(self.X)
        return self.model

    def predict(self, model):
        """
        Generates the centroid coordinates (cluster centers).
        :param model: The KMeans model
        :return: Clusters centers
        """
        model_predict = model.predict(self.X)
        centroids = model.cluster_centers_
        return model_predict, centroids

    def visualize(self, model_predict, centroids, size, s, cmap_s, centroids_size):
        """
        Displays the clusters on a scatterplot using two sets of elements (model_predict and centroids).
        :param model_predict:The KMeans model
        :param centroids: Clusters centers
        :param figsize: A tuple of values for a figure
        :param s: scalar marker size
        :param cmap_s: Color Map
        :param centroids_size: centroids marker size
        :return: A pyplot instance
        """
        plt.figure(figsize=size)
        plt.scatter(self.X[:, 0], self.X[:, 1], c=model_predict, s=s, cmap=cmap_s)
        plt.scatter(centroids[:, 0], centroids[:, 1], c='black', s=centroids_size, alpha=1)
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


