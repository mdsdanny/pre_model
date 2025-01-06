"""
Represents a Model
"""
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn import metrics
from core import DataFrame

class Model:
    #TODO create sub class model file with only pandas
    def __init__(self, csv_file=None, sep=None):
        """
        Initializes a DataFrame instance
        :param csv_file: CSV file to parse
        :param sep: a separator
        """
        if csv_file:
            self.df = DataFrame(csv_file, sep)
        self.X = []
        self.y = []
        self.model = None

    def render(self, mpyplot, x_label, y_label, size=None):
        #TODO render image to disk
        return None

    def set_x(self, variables):
        """
        The X array contains the independent variables.
        :param variables: independent variables
        :return: None
        """
        self.X = self.df.data_frame()[variables]

    def set_x_scaled(self, scaled):
        """
        The X array contains the independent variables.
        :param scaled: independent variables
        :return: None
        """
        self.X = scaled

    def set_non_x(self, variable):
        """
        The X array contains the independent variables.
        :param variables: independent variables
        :return: None
        """
        self.X = self.df.data_frame().drop(variable, axis=1)

    def set_y(self, variable):
        """
        The y contains the dependent variable.
        :param variable: dependent variable
        :return:
        """
        self.y = self.df.data_frame()[variable]

    def perform_split(self):
        """
        Shuffle and subdivides the data into training and test data. Using a standard of 70/30 % split
        with a random seed number of 10.
        :return:
        """
        return train_test_split(self.X, self.y, test_size=0.3, random_state=10, shuffle=True)

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

    def heatmap(self, fn, cmap=None):
        """
        Displays relationships between variables. Is a matrix of structure rows and columns representing values in color.
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
        sns.heatmap(self.df.corr(), annot=True, cmap=cmap)


    def pairplot(self, fn, columns=None):
        """
        Displays patterns between two variables. Takes the form of a 2D or 3D grid of plots of variables against other variables of the data frame.
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
        sns.pairplot(self.df.data_frame(), vars=columns)

    def distplot(self, fn, variable):
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
        sns.distplot(self.df.data_frame()[variable], kde=True, hist=0)

    def drop_variables(self, variables):
        """
        Removes fully the given variables from the dataframe.
        :param variables: variables to be removed.
        :return:
        """
        self.df.drop_columns(variables)

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