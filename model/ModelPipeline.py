from model.Train import Train
from model.evaluation.EvaluationPipeline import EvaluationPipeline
from model.OutputModule import Output
from joblib import dump


class ModelPipeline:
    """
    Pipeline in charge of running all the train/test/evaluation procedure.
    """

    def __init__(self, parameters):
        """
        Initialize a new instance of ModelPipeline

        Args:
            parameters (dictionary): Set of parameters that contain all the needed information for
            running the pipeline.
        """
        self.parameters = parameters

    def run(self):
        """
        Trains the model for finally testing and generate a file that will contain all the information related to
        the process.
        """
        # We instantiate the training pipeline and we train the model
        training_pipeline = Train(self.parameters)
        model, best_params = training_pipeline.train()

        self.parameters['best_params'] = best_params

        # We collect all the evaluation metrics from the trained model
        evaluation_pipeline = EvaluationPipeline(model, self.parameters)
        evaluation_results = evaluation_pipeline.evaluate()

        self.parameters['evaluation_results'] = evaluation_results

        # We generate a file containing all the details of this run
        output_generator = Output(self.parameters)

        dump(model, self.parameters['output_path'] + self.parameters['model'] + 'model.joblib')

        return output_generator.generate_dataframe()
