from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from core import Model

class GridSearch(Model):

    def new_model(self, hyperparameters):
        """
        Creates and assigns a new GridSearchCV.
        :return: None
        """
        self.model = GridSearchCV(SVC(), hyperparameters)

    def bestParams(self):
        """
        #TODO concept
        :return:
        """
        bp = self.model.best_params_
        print(f'bp is {bp}')
        return bp