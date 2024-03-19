from model.models.Model import Model
from sklearn.linear_model import ElasticNet as ElasticNetModel


class ElasticNet(Model):
    def __init__(self, parameters):
        """
        Initialize a new instance of LogisticRegression which is a subclass of the Model class which is also
        instantiated inside this constructor.

        Args:
            parameters (dict): contains all the needed information, from training data to hyperparameters grid.

        """
        self.parameters = parameters
        if 'parameters_grid' not in self.parameters or not self.parameters['parameters_grid']:
            self.parameters['parameters_grid'] = self.param_grid

        Model.__init__(self, parameters, ElasticNetModel(random_state=self.parameters['seed']))

    def train(self):
        """
        Used for training the model, it just calls to the method in the superclass.
        """
        return super().train()
    