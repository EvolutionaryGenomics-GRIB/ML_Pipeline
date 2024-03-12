from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler


class ClassBalancer:
    """
    Balances the data according to the target variable that we are studying, different methods will be implemented
    including oversampling and undersampling techniques.
    """

    def __init__(self, dataframe, target, parameters):
        """
        Initialization of ClassBalancer class

        Args:
            dataframe (dataframe): Data
            target (string): target variable name.
            parameters (dict): parameters dictionary containing all the needed information.
        """
        self.dataframe = dataframe
        self.target = target
        self.parameters = parameters

    def balance_classes(self):
        """
        Depending on the balancing technique calls the implementation of it.

        Returns:
            Returns the balanced dataframe.
        """
        if self.parameters['class_balancer'] == 'smote':
            return self.transform(SMOTE(random_state=self.parameters['seed']))
        elif self.parameters['class_balancer'] == 'random_oversampling':
            return self.transform(RandomOverSampler(random_state=self.parameters['seed']))
        elif self.parameters['class_balancer'] == 'random_undersampling':
            return self.transform(RandomUnderSampler(random_state=self.parameters['seed']))

        else:
            return self.dataframe

    def transform(self, balancer):
        X_resampled, y_resampled = balancer.fit_resample(self.dataframe, self.target)

        dataframe_resampled = X_resampled.copy()

        return dataframe_resampled, y_resampled
