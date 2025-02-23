"""
Principal Component Analysis (PCA) is a dimension reduction technique.
Also, known as General factor analysis. Is useful for dramatically reducing data complexity
and visualizing data in fewer dimensions, allowing to preserve as much of the original variation
as possible.
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from core import PreModel
from scrubbing import DataScrubbing


class Pca(PreModel):

    def __init__(self):
        super().__init__()
        self.df = None
        self.scaled = None
        self.transformed = None

    def scale(self, df):
        """
        Standardize features by using zero as the mean foar all variables and scaling to unit variance.
        Fit the function to the features contained in the dataframe and transforms those variables.
        :param df
        :return: None
        """
        self.df = df
        scaler = StandardScaler()
        scaler.fit(self.df)
        self.scaled = scaler.transform(self.df)

    def pca_transform(self, n_components):
        """
        Reshape the dataframe's features into a defined number of components.
        That best explain variability in the data.
        :param n_components: Number of components.
        """
        self.pre_model = PCA(n_components=n_components)
        self.pre_model.fit(self.scaled)
        self.transformed = self.pre_model.transform(self.scaled)

    def visualize(self, size, legend, colors, labels):
        """
        Visualizes the components of a combinations of variables.
        :param size: A tuple of values for a figure
        :param legend: The name of the dataframe column
        :param colors: A dictionary of RGB colors
        :param labels: A dictionary of labels
        :return: A pyplot instance
        """
        plt.figure(figsize=size)
        for t in np.unique(self.df[legend]):
            ix = np.where(self.df[legend] == t)
            plt.scatter(self.transformed[ix, 0], self.transformed[ix, 1], c=colors[t], label=labels[t])
        return plt

    def render(self, mpyplot, x_label, y_label, size=None):
        """
        It renders the matplotlib.pyplot final visualization.
        :param mpyplot: A matplotlib.pyplot instance
        :param x_label: Label of X axis
        :param y_label: Label of Y axis
        :param size: A tuple of values for a figure
        :return: None
        """
        mpyplot.xlabel(x_label)
        mpyplot.ylabel(y_label)
        mpyplot.legend()
        mpyplot.show()

