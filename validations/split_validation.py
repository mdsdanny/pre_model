"""
Split Validation splits the data into two separate sets. The first set is called the training
data and is used to build the prediction model. The second set is called the test data
and is kept in reserve and used to assess the accuracy of the model developed from the training data.
"""
from core.model import Model
from sklearn.model_selection import train_test_split

class SplitValidation(Model):

    def init_model(self):
        pass

    def split(self, df, x_variables, y_variable):
        """
        #TODO
        """
        # Step 4 Data Scrubbing
        self.set_x(df, x_variables)
        self.set_y(df, y_variable)
        # Step 6 Split Validation
        return self.perform_split()

    def split_scaled(self, x_variables, y_variable):
        """
        #TODO
        """
        # Step 4 Data Scrubbing
        self.set_x_scaled(x_variables)
        self.set_y_scaled(y_variable)
        # Step 6 Split Validation
        return self.perform_split()

    def perform_split(self, test_size=0.3, random_state=10, shuffle=True):
        """
        Shuffle and subdivides the data into training and test data. Using a standard of 70/30 % split
        with a random seed number of 10.
        :return:
        """
        return train_test_split(self.X, self.y, test_size=test_size, random_state=random_state, shuffle=shuffle)

