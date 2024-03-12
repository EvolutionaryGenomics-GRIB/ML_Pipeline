from model.models.Model import Model
from sklearn.ensemble import RandomForestClassifier


class RandomForest(Model):

    param_grid = {
        'n_estimators': [100],  # Number of trees in the forest
        'criterion': ['gini'],  # Splitting criterion (impurity measure)
        'max_depth': [None],  # Maximum depth of individual trees
        'min_samples_split': [2],  # Minimum number of samples required to split a node
        'min_samples_leaf': [12],  # Minimum number of samples required to be at a leaf node
        'max_features': ["sqrt"],  # Number of features to consider when splitting
        'bootstrap': [True],  # Whether to bootstrap samples when building trees
        'oob_score': [True],  # Whether to use out-of-bag samples to estimate generalization error
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

        Model.__init__(self, parameters, RandomForestClassifier(random_state=self.parameters['seed']))

    def train(self):
        """
        Used for training the model, it just calls to the method in the superclass.
        """
        return super().train()
