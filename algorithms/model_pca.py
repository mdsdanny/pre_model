"""
Principal Component Analysis (PCA) is a dimension reduction technique.
Also, know as General factor analysis.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from algorithms import Algorithm


class ModelPCA(Algorithm):

    def __init__(self, csv_file, drop_columns):
        """
        Initializes a csv file and non-numerical columns for removal.
        :param csv_file: File in local system
        :param drop_columns: non-numerical column names for removal
        """
        super().__init__()
        self.csv_file = csv_file
        self.drop_columns = drop_columns
        self.df = None

    def read_file(self):
        """
        Reads the csv file as dataframe and removes non-numerical columns.
        :return: None 
        """
        self.df = pd.read_csv(self.csv_file)
        for c in self.drop_columns:
            del self.df[c]

    def scale(self):
        """
        Standardize features by using zero as the mean foar all variables and scaling to unit variance.
        Fit the function to the features contained in the dataframe and transforms those variables.
        :return: Scaled data
        """
        scaler = StandardScaler()
        scaler.fit(self.df)
        return scaler.transform(self.df)

    def pca(self, scaled_data, n_components):
        """
        Reshape the dataframe's features into a defined number of components.
        That best explain variability in the data.
        :param scaled_data: Scaled data
        :param n_components: Number of components.
        :return: The transform data into components.
        """
        pca = PCA(n_components=n_components)
        pca.fit(scaled_data)
        return pca.transform(scaled_data)

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
        for t in np.unique(self.df[legend]):
            ix = np.where(self.df[legend] == t)
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

