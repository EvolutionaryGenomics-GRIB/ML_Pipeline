import csv

import pandas as pd


class Output:
    """
    Class that represents the Output of the program, that will allocate all the information related to it.
    """

    def __init__(self, parameters):
        """
        Initialize a new instance of Output

        Args:
            parameters (dictionary): Set of parameters and results that have been collected during execution.
        """
        self.parameters = parameters

    def generate_dataframe(self):
        """
        Method used for generating the dataframe that will store the output of both pipelines.

        Returns:
            A pandas dataframe containing the output of a particular run of the pipeline.
        """
        columns = []
        values = []

        # Removal of all the parameters that we do not want to output.
        del self.parameters['X_train']
        del self.parameters['X_test']
        del self.parameters['y_train']
        del self.parameters['y_test']
        del self.parameters['dataframe']
        del self.parameters['parameters_grid']

        # Round metrics values to 4 decimals
        for key, value in self.parameters['evaluation_results'].items():
            self.parameters['evaluation_results'][key] = round(value, 4)

        for key, value in self.parameters.items():
            if isinstance(value, dict) and key is not 'feature_importances':
                for sub_key, sub_value in value.items():
                    columns.append(sub_key)
                    values.append(sub_value)
            else:
                columns.append(key)
                values.append(value)

        return pd.DataFrame([values], columns=columns)
