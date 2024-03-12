from model.models.GradientDescent import GradientDescent
from model.models.LogisticRegression import LogisticRegression
from model.models.RBF_SVM import RBF_SVM
from model.models.RandomForest import RandomForest
from model.models.XGBoost import XGBoost


class Train:
    """
    Object that represents the training procedure that will store all the needed information for it.
    """

    def __init__(self, parameters):
        """
        Initialize a new instance of Train class.

        Args:
            parameters (dictionary): contains all the needed parameters.
        """
        self.parameters = parameters

    def train(self):
        """
        Performs the training step which varies depending on the type of model that is used.

        Returns:
            model (object): trained model.
            best_params (dictionary): dictionary containing the best hyperparameters for this model.

        """
        if self.parameters['model'] == 'logistic_regression':
            model = LogisticRegression(self.parameters)
            model = model.train()
            best_params = model.best_params_

        elif self.parameters['model'] == 'random_forest':
            model = RandomForest(self.parameters)
            model = model.train()
            best_params = model.best_params_

        elif self.parameters['model'] == 'xgboost':
            model = XGBoost(self.parameters)
            model = model.train()
            best_params = model.best_params_

        elif self.parameters['model'] == 'rbf_svm':
            model = RBF_SVM(self.parameters)
            model = model.train()
            best_params = model.best_params_

        elif self.parameters['model'] == 'gradient_descent':
            model = GradientDescent(self.parameters)
            model = model.train()
            best_params = model.best_params_

        return model, best_params
