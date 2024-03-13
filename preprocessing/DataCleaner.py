import pandas as pd

from preprocessing.Imputer import Imputer


class DataCleaner:
    """
    Standardization of columns, duplicate checking, data formatting, imputation, and outlier removal are performed
    in this Class.
    """

    def __init__(self, dataframe, parameters):
        """
        Initialize a new instance of DataCleaner

        Args:
            dataframe (dataframe): Data.
            parameters (dictionary): parameters dictionary that contains all the information related to the process.
        """
        # We attempt to convert columns to its proper type
        self.dataframe = dataframe.infer_objects()
        self.parameters = parameters

        self.imputer = Imputer()



    def clean_data(self):
        """
        This is the ML_pipeline_0.01 function of the class that will perform the data cleaning procedure.

        Returns:
            The cleaned dataframe.
        """
        if 'imputer' in self.parameters and self.parameters['imputer']:
            self.dataframe = self.impute(self.parameters['imputer'])

        return self.dataframe

    def remove_duplicates(self):
        """
        Remove all the duplicated rows from the dataframe.
        """
        self.dataframe = self.dataframe.drop_duplicates()

    def impute(self, imputation_technique):
        """
        Calls the imputer in order to perform the imputation.

        Args:
            imputation_technique (string): Technique to be used.

        Returns:
            The imputed dataframe.
        """
        return self.imputer.impute(self.dataframe, imputation_technique)
