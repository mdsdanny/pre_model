from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from core import Model

class GridSearch(Model):

    def init_model(self):
        """
        Creates and assigns a new GridSearchCV.
        :return: None
        """
        self.model = GridSearchCV(SVC(), {'C': [10, 25, 50], 'gamma': [0.001, 0.0001, 0.00001]})

    def bestParams(self):
        """
        #TODO concept
        :return:
        """
        bp = self.model.best_params_
        print(f'bp is {bp}')
        return bp