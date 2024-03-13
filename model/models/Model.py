from sklearn.model_selection import GridSearchCV
import numpy as np


class Model:
    """
    Superclass of all the ML models, represents the concept of Model which is meant to store all the methods
    commonly used for training the models.
    """

    def __init__(self, parameters, model):
        """
        Initialize a new instance of Model.

        Args:
            parameters (dict): contains all the needed information, from training data to hyperparameters grid.
            model (object): Instantiated Machine Learning model.
        """
        self.parameters = parameters
        self.X = self.parameters['X_train']
        self.y = self.parameters['y_train']
        self.model = model
        self.param_grid = self.parameters['parameters_grid']
        self.best_score = 0.0
        self.enable_grid_modification = self.parameters['enable_parameter_search']

    def train(self):
        """
        Method that recursively trains the model modifying the hyperparameters dynamically until the current
        score do not improve the old one.

        Returns:
            The trained grid search that contains the feature importances, best score, the best hyperparameters,
            among other important parameters.
        """
        min_class_samples = np.min(np.bincount(self.y))
        n_splits = min(15, min_class_samples)

        grid_search = GridSearchCV(self.model, self.param_grid, cv=n_splits, scoring='roc_auc', n_jobs=None, verbose=1)
        grid_search.fit(self.X, self.y)

        if not self.enable_grid_modification:
            return grid_search
        # New grid search is not improving the score
        elif round(grid_search.best_score_, 2) < round(self.best_score, 2):
            return self.grid_search
        else:
            self.grid_search = grid_search

            self.best_score = grid_search.best_score_
            self.best_parameters = grid_search.best_params_

            self.modify_grid_params()
            return self.train()

    def generate_interval(self, x, y, n):
        """
        Generates an interval between x and y [x,y] of size n

        Args:
            x (numeric): First element of the interval
            y (numeric): Second number of the interval
            n (int): Number of elements in the interval

        Returns:
            Returns a list with the interval
        """

        # Calculate the interval between a and b
        interval = abs(x - y)
        # Divide the interval into n equal parts
        step = interval / (n + 1)

        numbers = []
        for i in range(n):
            number = min(x, y) + (i + 1) * step
            numbers.append(number)

        return numbers

    def generate_previous_number(self, x):
        """
        Used to generate a previous number for the hyperparameter fine tune when the optimal value is the lowest
        one in the list.

        Args:
            x (numerical): Optimal value from the hyperparameter fine tune from where we will extract a previous
            number.

        Returns:
            The previous number generated from x
        """
        interval = abs(x) / 4
        prev = x - (3 * interval)

        return prev

    def generate_next_number(self, x):
        """
        Used to generate the next number for the hyperparameter fine tuning when the optimal value is the highest
        one in the list.

        Args:
            x (numerical): Optimal value from the hyperparameter fine tuning from where we will extract the next
            number.

        Returns:
            The next number generated from x
        """
        interval = abs(x) / 4
        prev = x + (3 * interval)

        return prev

    def modify_grid_params(self):
        """
        Method that modifies all numerical list of hyperparameters dynamically. It iterates through all the
        hyperparameters and taking into account the position of the optimal value it generates a new range
        of values that will be used in order to optimize even more the hyperparameters.
        """
        for parameter, value in self.best_parameters.items():
            if any(isinstance(value, cls) for cls in [int, float]) and len(self.param_grid[parameter]) > 1:
                parameter_values = self.param_grid[parameter]
                position = parameter_values.index(value)

                var_type = type(value)

                # First we contemplate the case where the optimal value is the last in the initial range
                if position == len(parameter_values) - 1:
                    new_top = self.generate_next_number(value)
                    new_bottom = parameter_values[position - 1]
                # Optimal value is the first element
                elif position == 0:
                    new_bottom = self.generate_previous_number(value)
                    new_top = parameter_values[position + 1]
                # Optimal value in the middle
                else:
                    new_bottom = parameter_values[position - 1]
                    new_top = parameter_values[position + 1]

                new_range_bottom = self.generate_interval(new_bottom, value, 2)
                new_range_top = self.generate_interval(new_top, value, 2)

                new_range_bottom = [var_type(x) for x in new_range_bottom]
                new_range_top = [var_type(x) for x in new_range_top]

                self.param_grid[parameter] = new_range_bottom + new_range_top
