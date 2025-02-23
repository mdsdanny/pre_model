"""
In supervised learning, Linear regression plots a straight line or plane
called the hyperplane that predicts the target value of data inputs by
determining the dependence between the dependent variable(y) and its
changing independent variables(X). In a p-dimensional space, a
hyperplane is a subspace equivalent to dimension p-1. In two-dimensional
space, a hyperplane is a one-dimensional subspace/flat line.
"""
from sklearn.linear_model import LinearRegression
from core import Model
from core import DataFrame
from scrubbing import DataScrubbing
from validations import SplitValidation


class Linearregression(Model):

    def init_model(self):
        """
        Creates and assigns a new Linear Regression model.
        :return: None
        """
        self.model = LinearRegression()

    '''
    def results(self, df, columns):
        """
        Creates a table for each row of the independent variable and a specific column.
        :param columns: A column's
        :return:
        """

        return df.make_results(self.coef(), self.X.columns, columns=columns)
    '''
