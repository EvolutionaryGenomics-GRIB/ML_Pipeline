import pandas as pd
from preprocessing.DataCleaner import DataCleaner
from preprocessing.ClassBalancer import ClassBalancer
from preprocessing.DataTransformer import DataTransformer
from preprocessing.FeatureSelector import FeatureSelector
from sklearn.model_selection import train_test_split
import random


class PreprocessingPipeline:
    """
    This is the pipeline that will be in charge of performing all the preprocessing steps.
    """

    def __init__(self, dataframe, parameters):
        """
        Initialize a new instance of the Preprocessing pipepiline.

        Args:
            dataframe (dataframe): Data.
            parameters (dictionary): parameters dictionary that contain all the information related to the process.
        """
        self.categorical_data = None
        self.numerical_data = None
        self.dataframe = dataframe
        self.parameters = parameters

    def split_by_data_type(self):
        """
        Splits the data in categorical and numerical and assigns it to its respective class parameters
        """
        categorical_variables = self.dataframe.select_dtypes(include=['object', 'category']).columns.to_list()
        numerical_variables = list(filter(lambda x: x not in categorical_variables, self.dataframe.columns))

        self.numerical_data = self.dataframe[numerical_variables]
        self.categorical_data = self.dataframe[categorical_variables]

    def run(self):
        """
        Main method of the pipeline, where it goes through all the preprocessing steps.

        Returns:
            The preprocessed dataframe.
        """
        # Keep wanted features only
        if 'features' in self.parameters and len(self.parameters['features']) >= 1:
            variables = self.parameters['features'] + [self.parameters['target']]
            try:
                self.dataframe = self.dataframe[variables]
            except Exception as e:
                raise Exception("Some variables are not in the dataframe: " + str(e))

        # Split in numerical and categorical
        self.split_by_data_type()

        # Data cleaning
        data_cleaner = DataCleaner(self.numerical_data, self.parameters)
        self.numerical_data = data_cleaner.clean_data()

        # Data transforming
        data_transformer = DataTransformer(self.numerical_data, self.categorical_data, self.parameters)
        self.numerical_data, self.categorical_data = data_transformer.transform_data()

        ### FEATURE EXTRACTION ###

        # TODO

        ### FEATURE COMBINATION ###

        # TODO

        # Concatenate both dataframes
        index = self.dataframe.index
        self.dataframe = pd.concat(
            [self.numerical_data.reset_index(drop=True), self.categorical_data.reset_index(drop=True)], axis=1)

        self.dataframe.index = index
        self.dataframe = self.dataframe.dropna()

        # Feature selection
        if 'feature_selector' in self.parameters and self.parameters['feature_selector'] \
                and 'num_features' in self.parameters and self.parameters['num_features'] > 0:
            feature_selector = FeatureSelector(self.dataframe, self.parameters['target'],
                                               self.parameters['feature_selector'],
                                               self.parameters['num_features'])
            self.dataframe = feature_selector.select_features()

        if 'seed' not in self.parameters or not self.parameters['seed']:
            self.parameters['seed'] = random.randint(1, 999999)

        if self.parameters['evaluation_technique'] == 'train_test':
            X_train, X_test, y_train, y_test = train_test_split(self.dataframe.drop(self.parameters['target'], axis=1),
                                                                self.dataframe[self.parameters['target']],
                                                                test_size=0.3,
                                                                random_state=self.parameters['seed'])
        else:
            X_train = self.dataframe.drop(self.parameters['target'], axis=1)
            y_train = self.dataframe[self.parameters['target']]

            X_test = 0
            y_test = 0

        if 'class_balancer' in self.parameters and self.parameters['class_balancer']:
            class_balancer = ClassBalancer(X_train, y_train,
                                           self.parameters['class_balancer'], self.parameters['seed'])
            X_train, y_train = class_balancer.balance_classes()

        self.parameters['dataframe'] = self.dataframe
        self.parameters['X_train'] = X_train
        self.parameters['X_test'] = X_test
        self.parameters['y_train'] = y_train
        self.parameters['y_test'] = y_test

        self.parameters['sample_size'] = self.dataframe.shape[0]

        return self.parameters
