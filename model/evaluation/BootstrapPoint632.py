import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, auc, roc_curve, roc_auc_score

from model.evaluation.EvaluateModel import EvaluateModel
from model.evaluation.bootstrap_point632 import bootstrap_point632_score


class BootstrapPoint632(EvaluateModel):
    """
    Class including all methods for Botstrap .632+ method.
    """
    def __init__(self, parameters):
        """
          Initialize a new instance of the class

          Args:
              parameters (dictionary): dictionary containing all the needed parameters.

        """
        self.runs = parameters['bootstrap_runs']
        self.parameters = parameters

    def evaluate(self):
        """
        Performs bootstrap .632+ over all the existing ML metrics.

        Returns:
            The dictionary containing all the different metrics.
        """
        # callable_metrics = {'auc': roc_auc_score, 'accuracy': accuracy_score, 'precision': precision_score,
        #                     'recall': recall_score, 'f1': f1_score}
        callable_metrics = {'auc': roc_auc_score}
        metrics = {'auc': 0, 'accuracy': 0, 'precision': 0, 'recall': 0, 'f1': 0}

        X = self.parameters['X_train']
        y = self.parameters['y_train']

        model, _ = self.instantiate_model(X, y, model_type=self.parameters['model'],
                                           params=self.parameters['best_params'])

        feature_names = self.parameters['X_train'].columns.tolist()

        for k, v in callable_metrics.items():
            scores, feature_importances = bootstrap_point632_score(model, X, y, self.parameters['model'], feature_names, n_splits=self.runs, method='.632+', scoring_func=v,
                                              predict_proba=(k == 'auc' or k == 'roc'))

            if k == 'auc':
                self.parameters['feature_importances'] = feature_importances

            score = np.mean(scores)
            metrics[k] = score

        return metrics
