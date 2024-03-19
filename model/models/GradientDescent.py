from model.models.Model import Model
from sklearn.linear_model import SGDClassifier


class GradientDescent(Model):
    param_grid = {
        'loss': ['log_loss'],  # The loss function to be optimized
        'penalty': ['l2'],  # The regularization term to be applied ('l2', 'l1', or 'elasticnet')
        'alpha': [0.0001],  # Constant that multiplies the regularization term
        'l1_ratio': [0.15], # The mix ratio between 'l1' and 'l2' regularization for 'elasticnet' penalty
        'learning_rate': ['optimal'],  # The learning rate schedule
        'eta0': [0.0],  # The initial learning rate for 'constant', 'invscaling', and 'adaptive' schedules
        'max_iter': [1000]  # The maximum number of iterations
    }

    def __init__(self, parameters):
        """
        Initialize a new instance of XGBoost which is a subclass of the Model class which is also
        instantiated inside this constructor.

        Args:
            parameters (dict): contains all the needed information, from training data to hyperparameters grid.

        """
        self.parameters = parameters
        if 'parameters_grid' not in self.parameters or not self.parameters['parameters_grid']:
            self.parameters['parameters_grid'] = self.param_grid

        Model.__init__(self, parameters, SGDClassifier(random_state=self.parameters['seed']))

    def train(self):
        """
        Used for training the model, it just calls to the method in the superclass.
        """
        return super().train()
