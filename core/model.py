"""
Represents a Model
"""
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from sklearn import metrics
from abc import ABC, abstractclassmethod, abstractmethod


class Model(ABC):
    def __init__(self):
        """
        Initializes a DataFrame instance
        """
        self.X = []
        self.y = []
        self.model = None

    def render(self, mpyplot, x_label, y_label, size=None):
        #TODO render image to disk
        return None

    def set_x(self, df, variables):
        """
        The X array contains the independent variables.
        :param df:
        :param variables: independent variables
        :return: None
        """
        self.X = df[variables]

    def set_x_scaled(self, scaled):
        """
        The X array contains the independent variables.
        :param scaled: independent variables
        :return: None
        """
        self.X = scaled

    def set_non_x(self,  df, variable):
        """
        The X array contains the independent variables.
        :param df:
        :param variable: independent variable
        :return: None
        """
        self.X = df.drop(variable, axis=1)

    def set_y(self, df, variable):
        """
        The y contains the dependent variable.
        :param df:
        :param variable: dependent variable
        :return:
        """
        self.y = df[variable]

    def fit(self, X_train, y_train):
        """
        Links the training data to the model.
        :param X_train: X variables trains data.
        :param y_train: y variable train data.
        :return: None
        """
        self.model.fit(X_train, y_train)

    def predict(self, independents):
        """
        Predicts a value for an individual property given an array of independents variables.
        :param independents: a given array that represents the independents variables.
        :return:
        """
        return self.model.predict(independents)

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

    @abstractmethod
    def init_model(self):
        pass

    def heatmap(self, corr, fn, cmap=None):
        """
        Displays relationships between variables. Is a matrix of structure rows and columns representing values in color.
        :param corr:
        :param fn: Figure's number
        :param cmap: a Layout of colors
        :return: None
        """
        if cmap is None:
            cmap = 'cool'
        f = plt.figure(fn)
        t = f.canvas.new_timer(interval=100000)
        t.start()
        t.add_callback(close_figure)
        plt.title("heatmap")
        sns.heatmap(corr, annot=True, cmap=cmap)


    def pairplot(self, df, fn, columns=None):
        """
        Displays patterns between two variables. Takes the form of a 2D or 3D grid of plots of variables against other variables of the data frame.
        :param df:
        :param fn: Figure's number
        :param columns: Names to evaluate
        :return: None
        """
        if columns is None:
            columns = []
        f = plt.figure(fn)
        plt.title("pairplot")
        t = f.canvas.new_timer(interval=100000)
        t.start()
        t.add_callback(close_figure)
        sns.pairplot(df, vars=columns)

    def distplot(self, df, fn, variable):
        """
        Displays behaviour of a variable in given frame.
        :param fn: Figure's number
        :param variable: Variable to evaluate
        :return: None
        """
        f = plt.figure(fn)
        plt.title("distplot")
        t = f.canvas.new_timer(interval=100000)
        t.start()
        t.add_callback(close_figure)
        sns.distplot(df[variable], kde=True, hist=0)



    @staticmethod
    def display():
        plt.show()

    @staticmethod
    def generate_predictions(test, model_predict):
        print(confusion_matrix(test, model_predict))
        print(classification_report(test, model_predict))

    @staticmethod
    def mean_absolute_error(y_test, prediction):
        """
        Compares the difference between the models expected value for a test set and the test set predictions.
        :param y_test: the y array results
        :param prediction:
        :return:
        """
        return metrics.mean_absolute_error(y_test, prediction)

def close_figure():
    plt.close()