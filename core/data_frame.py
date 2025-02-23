"""
Represents a pandas dataframe
"""
import pandas as pd

class DataFrame:

    def __init__(self, csv_file, sep=None, df=None):
        """
        Initializes a DataFrame
        :param csv_file:
        :param sep: a separator
        """
        self.df = df
        self.csv_file = csv_file
        if not df:
            self.read_file(sep)

    def read_file(self, sep):
        """
        Reads the csv file as a pandas dataframe.
        :param sep: a separator
        :return: None
        """
        if sep is None:
            self.df = pd.read_csv(self.csv_file)
        else:
            self.df = pd.read_csv(self.csv_file, sep=sep)

    def data_frame(self):
        """
        Gets the current data frame instance.
        :return: a pandas data frame
        """
        return self.df

    def drop_columns(self, columns):
        """
        Drops given column names.
        :param columns: Columns Names for removal
        :return: None
        """
        for c in columns:
            del self.df[c]

    def corr(self, numeric_only=True):
        """
        Compute pairwise correlation of columns, excluding NA/null values.
        A degree to which a pair of variables are linearly related.
        :return: corr function
        """
        return self.df.corr(numeric_only=numeric_only)

    def corr_variables(self, variable, another_variable):
        """
        Compute pairwise correlation of variables.
        :param variable: one variable
        :param another_variable: another variable
        :return: corr function
        """
        return self.df[variable].corr(self.df[another_variable])

    def get_dummies(self, columns, inplace=False):
        """
        Re-express categorical variables(as strings) as a numeric categorizer(like binary form). By a common technique One-hot encoding .
        :param columns: for applying one-hot encoding
        :param inplace: if modifies current data
        :return:
        """
        df_one = pd.get_dummies(self.df, columns = columns, drop_first = True)
        if inplace:
            self.df = df_one
            return self.df
        return df_one

    def null_sum(self):
        """
        Overviews column's missing values.
        :return: the number of null values per column
        """
        s = self.df.isnull().sum()
        print(f'sum is {s}')
        return s

    def fillna_average(self, column):
        """
        Fills the null column values with the average value of dataset for that variable.
        :param column: Null values column
        :return:
        """
        return self.df[column].fillna((self.df[column].mean()), inplace=True)

    def fillna_common(self, column):
        """
        Fills the null column values with the most common value of dataset for that variable.
        :param column: Null values column
        :return:
        """
        return self.df[column].fillna((self.df[column].mode()), inplace=True)

    def fillna_value(self, column, value):
        """
        Fills the null column values with given value.
        :param column: Null values column
        :param value: value to set
        :return:
        """
        return self.df[column].fillna(value)

    def dropna(self):
        """
        Automatically removes columns or rows that contains missing values on a case-by-case basis.
        :return: the DataFrame
        """
        #thresh=None
        return self.df.dropna(axis=0, how='any', subset=None, inplace=True)

    def make_results(self, coef, X, columns):
        """
        #TODO
        :param coef:
        :param X:
        :param columns:
        :return:
        """
        return pd.DataFrame(coef, X, columns)


