"""
Exploratory Data Analysis known as EDA. Involves summarizing the salient
characteristics of a dataset in preparation for further processing and analysis.
This includes, understanding the shape and distribution of the data, scanning
for missing values, learning which features are most relevant based on correlation,
and familiarizing yourself with the overall contents of the dataset.
"""
from core import DataFrame

class ExploratoryDataAnalysis(DataFrame):


    def head(self, n=10):
        """
        Previews the first n rows of the dataframe.
        :param n:
        :return: The dataframe n first rows
        """
        return self.df.head(n)

    def tail(self, n=10):
        """
        Previews the last n rows of the dataframe.
        :param n:
        :return: The dataframe n last rows
        """
        return self.df.tail(n)

    def shape(self):
        """
        Previews the number of rows and number of columns
        :return: A tuple of number of rows and number of columns
        """
        s = self.df.shape
        print(f'rows {s[0]}, columns {s[1]}')
        return s

    def row(self, index=None):
        """
        Retrieves the row indexed at position.
        :param index: The index to find
        :return: row as a dictionary
        """
        if index is None:
            index = -1
        row = self.df.iloc[index]

        for k, v in row.items():
            print(k, v)
        return row

    def columns(self):
        """
        Previews the columns of the dataframe
        :return: The columns names
        """
        names = self.df.columns
        for c in names:
            print(c)
        return names


    def describe(self, include=None):
        """
        Generates a summary (Statistics) of the dataframe's mean, standard deviation and IQR (interquartile range) values.
        :param include:
        :return: the summary
        """
        if include is not None:
            return self.df.describe(include=include)
        return self.df.describe()



