"""
Data Scrubbing is an umbrella term for manipulating data in preparation for analysis.
Some algorithms, for example, don't recognize specific data types or return an error message.
in response to missing values or non-numeric input. Variables too, may need to be scaled to size
or converted to a more compatible data type. Also, duplicate information, redundant variables,
and error in the data.
"""
import matplotlib.pyplot as plt

from eda import ExploratoryDataAnalysis


class DataScrubbing(ExploratoryDataAnalysis):

    def drop_variables(self, variables):
        """
        Removes fully the given variables from the dataframe.
        :param variables: variables to be removed.
        :return:
        """
        self.drop_columns(variables)


def display():
    plt.show()

def close_figure():
    plt.close()


