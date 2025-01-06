"""
Principal Component Analysis (PCA) is a dimension reduction technique.
Also, know as General factor analysis.
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from core.model import Model

class Pca(Model):

    def scale(self):
        """
        Standardize features by using zero as the mean foar all variables and scaling to unit variance.
        Fit the function to the features contained in the dataframe and transforms those variables.
        :return: Scaled data
        """
        scaler = StandardScaler()
        scaler.fit(self.df.data_frame())
        return scaler.transform(self.df.data_frame())

    def pca_transform(self, scaled_data, n_components):
        """
        Reshape the dataframe's features into a defined number of components.
        That best explain variability in the data.
        :param scaled_data: Scaled data
        :param n_components: Number of components.
        :return: The transform data into components.
        """
        self.model = PCA(n_components=n_components)
        self.model.fit(scaled_data)
        return self.model.transform(scaled_data)

    def visualize(self, scaled, size, legend, colors, labels):
        """
        Visualizes the components of a combinations of variables.
        :param scaled: A scaled data
        :param size: A tuple of values for a figure
        :param legend: The name of the dataframe column
        :param colors: A dictionary of RGB colors
        :param labels: A dictionary of labels
        :return: A pyplot instance
        """
        plt.figure(figsize=size)
        for t in np.unique(self.df.data_frame()[legend]):
            ix = np.where(self.df.data_frame()[legend] == t)
            plt.scatter(scaled[ix, 0], scaled[ix, 1], c=colors[t], label=labels[t])
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

