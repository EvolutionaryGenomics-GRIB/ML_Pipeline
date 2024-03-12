import random
from sklearn.model_selection import train_test_split
from model.evaluation.EvaluateModel import EvaluateModel
from collections import OrderedDict
import pandas as pd

class TrainTest(EvaluateModel):
    def __init__(self, parameters, model):
        """
        Initializes the Train/Test evaluation technique.

        Args:
            parameters (dict): the parameters dictionary containing all the needed values.
            model (sklearn model): the trained model.

        """
        self.runs = parameters['splitting_runs']
        self.X = parameters['X_test']
        self.y = parameters['y_test']
        self.parameters = parameters
        self.model = model

    def evaluate(self):
        """
        Performs evaluation following the train/test technique.

        Returns:
            The dictionary containing all the evaluation metrics.
        """
        y_pred = self.model.predict(self.X)
        y_pred_proba = self.model.predict_proba(self.X)[:, 1]

        metrics = self.compute_metrics(y_true=self.y, y_pred=y_pred, y_pred_proba=y_pred_proba)

        metrics_df = pd.DataFrame([metrics])

        feature_importances_df = pd.DataFrame()

        seeds = [random.randint(1, 99999) for _ in range(self.runs)]
        for seed in seeds:
            X_train, X_test, y_train, y_test = train_test_split(
                self.parameters['dataframe'].drop(self.parameters['target'], axis=1),
                self.parameters['dataframe'][self.parameters['target']],
                test_size=0.3, random_state=seed)

            model, feature_importances = self.instantiate_model(X_train, y_train, self.parameters['model'], self.parameters['best_params'])
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]

            metrics = self.compute_metrics(y_true=y_test, y_pred=y_pred, y_pred_proba=y_pred_proba)

            new_row = pd.DataFrame([metrics])

            feature_importances_df = pd.concat([feature_importances_df, pd.DataFrame([feature_importances])], axis=0)
            # Append the new row to the original DataFrame
            metrics_df = pd.concat([metrics_df, new_row], axis=0)

        roc_curves_df = metrics_df[['fpr', 'tpr']]

        # Calculate the average of each column
        average_values = metrics_df.drop(['fpr', 'tpr'], axis=1).mean()

        average_feature_importances = feature_importances_df.mean()

        # Convert the Series with average values to a dictionary
        average_dict = average_values.to_dict()

        average_feature_importances = average_feature_importances.to_dict()

        # Round the values to 3 decimals and convert to float
        rounded_feature_importances = {key: round(float(value), 3) for key, value in
                                       average_feature_importances.items()}

        # Sort the dictionary by values in descending order
        sorted_feature_importances = OrderedDict(
            sorted(rounded_feature_importances.items(), key=lambda x: x[1], reverse=True))

        self.parameters['feature_importances'] = sorted_feature_importances

        self.plot_roc(roc_curves_df, average_dict['auc'])

        return average_dict