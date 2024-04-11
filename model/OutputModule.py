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

    def generate_output_file(self):
        with open(self.parameters['output_path'] + 'output.txt', 'a') as file:
            file.write(f"Results for {self.parameters['model']}, N={self.parameters['sample_size']}:\n\n")
            file.write(f"Class balancer: {self.parameters['class_balancer']}, scaler: {self.parameters['scaler']}, encoder: {self.parameters['encoder']}, "
                       f"imputer: {self.parameters['imputer']} \n\n")
            file.write(f"Feature selector: {self.parameters['feature_selector']}, number of features to select: {self.parameters['num_features']} \n\n")
            file.write(f"Target variable: {self.parameters['target']}\n\n")

            feature_importances = ', '.join([f"('{key}', {value})" for key, value in self.parameters['feature_importances'].items()])

            file.write(f"Feature importances: {feature_importances} \n\n")
            file.write(f"Hyperparameters: {self.parameters['best_params']} \n\n")
            scores_dictionary = self.parameters['evaluation_results']
            file.write(f"Scores: Accuracy = {scores_dictionary['accuracy']}, Precision = {scores_dictionary['precision']}, "
                       f"Recall = {scores_dictionary['recall']}, F1 = {scores_dictionary['f1']}, AUC = {scores_dictionary['auc']}\n\n")
            file.write("-------------------------------------------\n\n")
