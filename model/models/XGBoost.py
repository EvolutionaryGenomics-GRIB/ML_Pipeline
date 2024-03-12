from model.models.Model import Model
import xgboost as xgb


class XGBoost(Model):
    # param_grid = {
    #     'learning_rate': [0.01, 0.1, 0.2, 0.3],
    #     'n_estimators': [50, 100, 200, 300],
    #     'max_depth': [3, 4, 5, 6],
    #     'min_child_weight': [1, 2, 3, 4],
    #     'subsample': [0.8, 0.9, 1.0],
    #     'colsample_bytree': [0.8, 0.9, 1.0],
    #     'gamma': [0, 0.1, 0.2, 0.3],
    #     'reg_alpha': [0, 0.1, 0.2, 0.3],
    #     'reg_lambda': [0, 0.1, 0.2, 0.3]
    # }

    param_grid = {
        'n_estimators': [50, 100],  # Number of boosting rounds (trees)
        'learning_rate': [0.15],  # Step size shrinkage used in each boosting round
        'max_depth': [4],  # Maximum depth of individual trees
        'min_child_weight': [5],  # Minimum sum of instance weight (hessian) needed in a child
        'subsample': [0.9],  # Fraction of samples used for fitting the trees
        'colsample_bytree': [0.9],  # Fraction of features used for building each tree
        'gamma': [0.1],  # Minimum loss reduction required to make a further partition on a leaf node
        'alpha': [1e-3],  # L1 regularization term on weights
        'lambda': [1e-3],  # L2 regularization term on weights
    }

    def __init__(self, parameters):
        """
        Initialize a new instance of XGBoost which is a subclass of the Model class which is also
        instantiated inside this constructor.

        Args:
            parameters (dict): contains all the needed information, from training data to hyperparameters grid.
        """
        self.parameters = parameters
        if 'parameters_grid' not in self.parameters:
            self.parameters['parameters_grid'] = self.param_grid

        Model.__init__(self, parameters, xgb.XGBClassifier(random_state=self.parameters['seed']))

    def train(self):
        """
        Used for training the model, it just calls to the method in the superclass.
        """
        return super().train()
