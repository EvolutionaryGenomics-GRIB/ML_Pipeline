from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression as LogisticRegressionModel
from model.models.Model import Model


class LogisticRegression(Model):
    """
    Subclass of Model that will represent the implementation of a Logistic Regression, containing all the needed
    information for the training of this model.
    """

    param_grid = {
        'C': [0.1, 0.75, 1, 10],  # Regularization parameter
        'penalty': ['l1'],  # Regularization type
        'solver': ['liblinear'],  # Solver algorithms
        'max_iter': [3000]  # Maximum number of iterations for the solver to converge
    }

    def __init__(self, parameters):
        """
        Initialize a new instance of LogisticRegression which is a subclass of the Model class which is also
        instantiated inside this constructor.

        Args:
            parameters (dict): contains all the needed information, from training data to hyperparameters grid.

        """
        self.parameters = parameters
        if 'parameters_grid' not in self.parameters:
            self.parameters['parameters_grid'] = self.param_grid

        Model.__init__(self, parameters, LogisticRegressionModel(random_state=self.parameters['seed']))

    def train(self):
        """
        Used for training the model, it just calls to the method in the superclass.
        """
        return super().train()




