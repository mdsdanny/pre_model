"""
In supervise learning, Linear regression plots a straight line or plane
called the hyperplane that predicts the target value of data inputs by
determining the dependence between the dependent variable(y) and its
changing independent variables(X). In a p-dimensional space, a
hyperplane is a subspace equivalent to dimension p-1. In two-dimensional
space, a hyperplane is a one-dimensional subspace/flat line.
"""
import pandas as pd
from sklearn.linear_model import LinearRegression
from core.model import Model


class Linearregression(Model):

    def results(self, columns):
        """
        Creates a table for each row of the independent variable and a specific column.
        :param columns: A column's
        :return:
        """
        return pd.DataFrame(self.coef(), self.X.columns, columns=columns)

    def new_model(self):
        """
        Creates and assigns a new Linear Regression model.
        :return: None
        """
        self.model = LinearRegression()

    def intercept(self):
        """
        Inspect the y-intercept of the model.
        :return: The y-intercept of the model
        """
        return self.model.intercept_

    def coef(self):
        """
        Inspects tje coefficients of the X independent variables.
        :return: coefficients of the X variables
        """
        st = str(self.model.coef_[0])
        st = st.replace('[', '')
        st = st.replace(']', '')
        return [float(s) for s in st.split()]



