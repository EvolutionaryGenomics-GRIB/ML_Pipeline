from scipy.interpolate import interp1d
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, auc, roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
import xgboost as xgb
import matplotlib.pyplot as plt
import numpy as np
from sklearn.svm import SVC
import matplotlib.backends.backend_pdf


def get_feature_importances(model, model_type, feature_names):
    """
    Retrieves the feature importances from a trained model.

    Args:
        model (object): trained model.

    Returns:
        A dictionary containing the feature importances.
    """
    feature_importances = {}

    if model_type == 'logistic_regression':
        coefficients = model.coef_[0]

        for i, feature in enumerate(feature_names):
            feature_importances[feature] = coefficients[i]

    elif model_type == 'rbf_svm':
        feature_importances = {}
    elif model_type == 'gradient_descent':
        coefficients = model.coef_[0]

        for i, feature in enumerate(feature_names):
            feature_importances[feature] = coefficients[i]

    else:
        coefficients = model.feature_importances_
        feature_importances = {}

        for i, feature in enumerate(feature_names):
            feature_importances[feature] = coefficients[i]

        return feature_importances

    return feature_importances

class EvaluateModel:
    """
    Superclass for the different evaluation techniques implemented in the pipeline, contains all the common methods
    """
    def instantiate_model(self, X_train, y_train, model_type, params):
        """
        Fits the model to the data passed as parameters and retrieves the feature importances

        Args:
            X_train (dataframe): Training data.
            y_train (array): target data.
            model_type (string): model to be instantiated.
            params (dictionary): dictionary containing the optimized hyperparameters of the model

        Returns:
            The fitted model and a dictionary containing the feature importances.
        """
        if model_type == 'xgboost':
            model = xgb.XGBClassifier(**params)
        elif model_type == 'random_forest':
            model = RandomForestClassifier(**params)
        elif model_type == 'logistic_regression':
            model = LogisticRegression(**params)
        elif model_type == 'rbf_svm':
            model = SVC(kernel='rbf', probability=True, **params)
        elif model_type == 'gradient_descent':
            model = SGDClassifier(**params)

        model.fit(X_train, y_train)

        feature_names = self.parameters['X_train'].columns.tolist()
        feature_importances = get_feature_importances(model, model_type, feature_names)

        return model, feature_importances

    def plot_roc(self, metrics_df, averaged_auc, overoptimistic_auc=0, overoptimistic_curve=[]):
        """
        Plots and saves the roc curve plot.

        Args:
            metrics_df (dataframe): The dataframe containing all the metrics.
            averaged_auc (float): the mean Area Under the Curve of the different runs.
            overoptimistic_auc (float): the overoptimistic AUC.
            overoptimistic_curve (list): roc curve for the overoptimistic run.

        """
        plt.figure(figsize=(8, 8))

        # Plots the overoptimistic curve if .632+ was selected as evaluation technique
        if overoptimistic_curve:
            plt.plot(overoptimistic_curve[0], overoptimistic_curve[1], color='navy', lw=2, linestyle='dotted',
                     label='Overoptimistic AUC: ' + str(round(overoptimistic_auc, 2)))

        tpr_values = metrics_df['tpr'].tolist()
        fpr_values = metrics_df['fpr'].tolist()

        # Find the minimum and maximum false positive rates across all runs
        min_fpr = min(np.min(run_fpr) for run_fpr in fpr_values)
        max_fpr = max(np.max(run_fpr) for run_fpr in fpr_values)

        # Choose a common set of mean_fpr values within the range
        mean_fpr = np.linspace(min_fpr, max_fpr, 100)

        # Interpolate individual ROC curves to the common set of mean_fpr values
        interp_tpr_values = [interp1d(fpr, tpr, kind='linear', fill_value='extrapolate')(mean_fpr) for fpr, tpr in
                             zip(fpr_values, tpr_values)]

        # Plots all the runs
        if not self.parameters['plot_mean_roc']:
            linestyle = '--'
            for index, row in metrics_df.iterrows():
                tpr = row['tpr']
                fpr = row['fpr']

                plt.plot(fpr, tpr, lw=1, color='grey')
        # Only the mean ROC curve is plotted
        else:
            linestyle = 'solid'

        mean_tpr = np.mean(interp_tpr_values, axis=0)

        if overoptimistic_auc != 0:
            averaged_auc = overoptimistic_auc - averaged_auc

        if self.parameters['roc_color']:
            color = self.parameters['roc_color']
        else:
            color = 'red'

        plt.plot(mean_fpr, mean_tpr, color=color, lw=2, linestyle=linestyle,
                 label='Averaged AUC: ' + str(round(averaged_auc, 2)))

        plt.xlabel('1 - Specificity (FPR)')
        plt.ylabel('Sensitivity (TPR)')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc='lower right')

        # Save the figure in pdf & png
        plt.savefig(self.parameters['output_path'] + self.parameters['model'] + '_roc.pdf')
        plt.savefig(self.parameters['output_path'] + self.parameters['model'] + '_roc.png')

    def compute_metrics(self, y_true, y_pred, y_pred_proba):
        """
        Once we get the predictions we compared against the true labels in order to extract all the common metrics
        used in Machine Learning: Accuracy, precision, recall, f1-score, AUC, and the confusion matrix.

        Args:
            y_true (array): true labels.
            y_pred (array): binary predictions.
            y_pred_proba (array): probabilistic predictions.

        Returns:
            The dictionary containing all the evaluation metrics.
        """
        # Initialize the dictionary to store evaluation results
        evaluation_results = {}

        # Calculate and store accuracy
        accuracy = accuracy_score(y_true, y_pred)
        evaluation_results['accuracy'] = accuracy

        # Calculate and store precision
        precision = precision_score(y_true, y_pred, average='weighted')
        evaluation_results['precision'] = precision

        # Calculate and store recall
        recall = recall_score(y_true, y_pred, average='weighted')
        evaluation_results['recall'] = recall

        # Calculate and store F1-score
        f1 = f1_score(y_true, y_pred, average='weighted')
        evaluation_results['f1'] = f1

        # # Calculate and store confusion matrix
        # conf_matrix = confusion_matrix(y_true, y_pred)
        # evaluation_results['confusion_matrix'] = conf_matrix

        # Compute ROC curve and ROC area for each class
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        evaluation_results['fpr'] = fpr
        evaluation_results['tpr'] = tpr

        roc_auc = auc(fpr, tpr)
        evaluation_results['auc'] = roc_auc

        return evaluation_results
