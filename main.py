import copy
import json
import os
import re
import sys
import time

import matplotlib.colors as mcolors
import pandas as pd

from model.ModelPipeline import ModelPipeline
from preprocessing.PreprocessingPipeline import PreprocessingPipeline


def exception_control(parameters):
    """
    Checks that all the given parameters have the correct type and are accepted by the framework. Raises an
    exception with a descriptive message in case anything is not correct.

    Args:
        parameters (dict): dictionary containing all the parameters of the model.

    """
    valid_parameters = {
        'target': {'type': str, 'error_msg': 'Parameter "{}" has to be a string'},
        'features': {'type': list, 'error_msg': 'Parameter "{}" has to be a list or the empty string'},
        'scaler': {'valid_values': ['min_max', 'z_score'],
                   'error_msg': 'The scaler is not available, the implemented scalers keys are: {}'},
        'encoder': {'valid_values': ['one_hot', 'target_encoding'],
                    'error_msg': 'The encoder is not available, the implemented encoders keys are: {}'},
        'class_balancer': {'valid_values': ['smote', 'random_oversampling', 'random_undersampling'],
                           'error_msg': 'The balancer is not available, the implemented balancer keys are: {}'},
        'evaluation_technique': {'valid_values': ['train_test', 'bootstrap', '.632+'],
                                 'error_msg': 'The evaluation is not available, the implemented evaluation keys are: {}'},
        'model': {'valid_values': ['logistic_regression', 'random_forest', 'xgboost', 'rbf_svm', 'gradient_descent'],
                  'error_msg': 'The model is not available, the implemented model keys are: {}'},
        'enable_parameter_search': {'type': bool, 'error_msg': 'Parameter "{}" has to be a boolean'},
        'splitting_runs': {'type': int, 'error_msg': 'Parameter "{}" has to be an integer'},
        'bootstrap_runs': {'type': int, 'error_msg': 'Parameter "{}" has to be an integer'},
        'output_path': {'type': str, 'error_msg': 'Parameter "{}" has to be a string'},
        'num_features': {'type': int, 'error_msg': 'Parameter "{}" has to be an integer'},
        'feature_selector': {'type': str, 'error_msg': 'Parameter "{}" has to be a string'},
        'parameters_grid': {'valid_types': [str, dict], 'error_msg': 'Parameter "{}" has to be a string'},
        'plot_mean_roc': {'type': bool, 'error_msg': 'Parameter "{}" has to be a boolean'},
        'roc_color': {'type': str, 'error_msg': 'The given color is not correct'},
        'test_size': {'type': float, 'error_msg': 'Parameter "{}" has to be a float'}
    }

    for k, v in parameters.items():
        if k not in valid_parameters:
            raise ValueError("This parameter name is not valid: {}".format(k))

        if 'type' in valid_parameters[k] and not isinstance(v, valid_parameters[k]['type']):
            raise TypeError(valid_parameters[k]['error_msg'].format(k))

        if 'valid_types' in valid_parameters[k] and not any(
                isinstance(v, t) for t in valid_parameters[k]['valid_types']):
            raise TypeError(valid_parameters[k]['error_msg'].format(k))

        if 'valid_values' in valid_parameters[k]:
            valid_values = valid_parameters[k]['valid_values']
            if v not in valid_values and v != "":
                raise ValueError(valid_parameters[k]['error_msg'].format(", ".join(valid_values)))

        if k == 'roc_color':
            if (v not in list(mcolors.CSS4_COLORS.keys())) and (not bool(re.match(r'^#[0-9a-fA-F]{6}$', v))):
                raise ValueError(valid_parameters[k]['error_msg'])


def generate_combinations(json_obj, current_combination, combinations_list):
    """
    In case any of the parameters has multiple options it generates all the possible combination between the different
    parameters so the pipeling is run as many times as combinations exist.

    Args:
        json_obj (dict): JSON object containing all the parameters.
        current_combination (dict): already converted dictionary from the JSON file.
        combinations_list (dict): list of converted dictionaries.
    """

    if not json_obj:
        combinations_list.append(current_combination)
        return

    key, value = json_obj.popitem()

    if ("features" not in key) and isinstance(value, list):
        for item in value:
            new_combination = copy.deepcopy(current_combination)
            new_combination[key] = item
            generate_combinations(json_obj.copy(), new_combination, combinations_list)
    else:
        current_combination[key] = value
        generate_combinations(json_obj, current_combination, combinations_list)


def create_output_folder(parameters):
    """
    Creates an output directory if the provided one is incorrect, in case the user makes multiple mistakes it
    creates different directories for every run.

    Args:
        parameters (dict): dicitonary contained all the parameters.
    """
    if not os.path.exists(parameters['output_path']):
        output_path = 'ML_Pipeline_Output'
    else:
        output_path = parameters['output_path'] + 'ML_Pipeline_Output'

    counter = 1
    while os.path.exists(output_path):
        output_path = f"ML_Pipeline_Output_{counter}/"
        counter += 1

    os.mkdir(output_path)
    parameters['output_path'] = output_path + '/'


def get_parameters_grid(parameters):
    try:
        with open(parameters['parameters_grid'], "r") as json_file:
            hyperparameters = json.load(json_file)
            return hyperparameters
    except ():
        print("Could not convert the parameters grid file to a dictionary, assigning default parameters.")
        return ""

def main():
    # Check if command-line arguments were provided
    if len(sys.argv) < 2:
        print("Usage: python script.py argument1 argument2")
        return

    if len(sys.argv) < 3:
        # Access command-line arguments
        arg1 = sys.argv[0]
        arg2 = sys.argv[1]
    else:
        arg1 = sys.argv[1]
        arg2 = sys.argv[2]

    # We need to convert our arg2 into a dictionary of parameters
    try:
        with open(arg2, "r") as json_file:
            parameters = json.load(json_file)
        if isinstance(parameters, dict):
            print("Converted dictionary:", parameters)
        else:
            print("The input is not a valid dictionary.")
    except (ValueError, SyntaxError):
        raise ValueError("Could not convert the parameters file to a dictionary.")

    # We read the data from our first parameter
    try:
        # We read our data from the path extracted from arg1
        dataframe = pd.read_csv(arg1)
    except (ValueError, SyntaxError):
        raise ValueError("Data has to be in a comma separated csv format")

    # Check if output dir exists and create it if not
    create_output_folder(parameters)

    if 'parameters_grid' in parameters and parameters['parameters_grid']:
        if not isinstance(parameters['parameters_grid'], dict) and os.path.exists(parameters['parameters_grid']):
            parameters['parameters_grid'] = get_parameters_grid(parameters)
        elif isinstance(parameters['model'], list) and len(parameters['model']) > 1:
            parameters['parameters_grid'] = ""

    combinations = []
    generate_combinations(parameters.copy(), {}, combinations)

    output_dataframe = pd.DataFrame()
    # We iterate through all the possible parameter combinations.
    for combination in combinations:
        # Raises an exception if any of the parameters is incorrect
        exception_control(combination)

        ### PREPROCESSING ###

        preprocessing_pipeline = PreprocessingPipeline(dataframe, combination)
        local_parameters = preprocessing_pipeline.run()

        ### MODEL TRAINING AND TESTING ###

        model_pipeline = ModelPipeline(local_parameters)
        aux = model_pipeline.run()
        output_dataframe = pd.concat([output_dataframe, aux], ignore_index=True)

    output_dataframe.T.to_csv(parameters['output_path'] + 'output.csv')


if __name__ == "__main__":
    start_time = time.time()
    main()
    print("--- %s seconds ---" % (time.time() - start_time))
