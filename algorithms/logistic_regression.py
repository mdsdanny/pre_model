"""
Logistic regression it involves predicting outcomes based on analyzing
quantitative relationships between variables. It accepts both continuous
and discrete variables as input and its output is qualitative; it predicts
a discrete class such as Yes/No or Customer/No customer.
"""
from sklearn.linear_model import LogisticRegression
from core import Model
from scrubbing import DataScrubbing
from validations import SplitValidation

class Logisticregression(Model):

    '''
    def results(self, columns):
        """
        Creates a table for each row of the independent variable and a specific column.
        :param columns: A column's
        :return:
        """
        return pd.DataFrame(self.coef(), self.X.columns, columns=columns)
    '''

    def init_model(self):
        """
        Creates and assigns a new Logistic Regression model.
        :return: None
        """
        self.model = LogisticRegression()

