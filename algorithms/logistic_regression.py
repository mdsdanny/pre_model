"""
Logistic regression it involves predicting outcomes based on analyzing
quantitative relationships between variables. It accepts both continuous
and discrete variables as input and its output is qualitative; it predicts
a discrete class such as Yes/No or Customer/No customer.
"""
import pandas as pd
from sklearn.linear_model import LogisticRegression
from core.model import Model

class Logisticregression(Model):

    def results(self, columns):
        """
        Creates a table for each row of the independent variable and a specific column.
        :param columns: A column's
        :return:
        """
        return pd.DataFrame(self.coef(), self.X.columns, columns=columns)

    def new_model(self):
        """
        Creates and assigns a new Logistic Regression model.
        :return: None
        """
        self.model = LogisticRegression()

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



