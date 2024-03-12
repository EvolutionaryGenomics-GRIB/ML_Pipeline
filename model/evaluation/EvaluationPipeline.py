from model.evaluation.BootstrapPoint632 import BootstrapPoint632
from model.evaluation.TrainTest import TrainTest


class EvaluationPipeline:
    """
    Pipeline for the evaluation of the ML model.
    """

    def __init__(self, model, parameters):
        """
        Initialize a new instance of the Pipeline

        Args:
            model: machine learning model to be tested.
            parameters (dictionary): dictionary containing all the needed parameters.

        """
        self.model = model
        self.parameters = parameters

    def evaluate(self):
        """
        Invokes the corresponding class according to the evaluation strategy.

        Returns:
            The dictionary containing all the different metrics.
        """

        if self.parameters['evaluation_technique'] == 'train_test':
            evaluation = TrainTest(self.parameters, self.model)
            return evaluation.evaluate()
        elif self.parameters['evaluation_technique'] == '.632+':
            evaluation = BootstrapPoint632(self.parameters)
            return evaluation.evaluate()
