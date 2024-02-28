from model.models.Model import Model
from sklearn.svm import SVC


class RBF_SVM(Model):

    param_grid = {
        'C': [0.1, 1],
        'gamma': [0.1, 1],
    }

    def __init__(self, parameters):
        """
        Initialize a new instance of LogisticRegression which is a subclass of the Model class which is also
        instantiated inside this constructor.

        Args:
            X (dataframe): Dataframe containing the training information for the model.
            y (array): Array containing the training target variable.
            seed (int): Seed to be used in the LogisticRegression
        """
        self.parameters = parameters
        if 'parameters_grid' not in self.parameters:
            self.parameters['parameters_grid'] = self.param_grid

        Model.__init__(self, parameters, SVC(kernel='rbf', probability=True, random_state=self.parameters['seed']))

    def train(self):
        """
        Used for training the model, it just calls to the method in the superclass.
        """
        return super().train()
