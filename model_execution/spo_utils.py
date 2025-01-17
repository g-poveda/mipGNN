import cplex
import os
import pickle
import itertools

# https://stackoverflow.com/a/40623158
def dict_product(dicts):
    """
    >>> list(dict_product(dict(number=[1,2], character='ab')))
    [{'character': 'a', 'number': 1},
     {'character': 'a', 'number': 2},
     {'character': 'b', 'number': 1},
     {'character': 'b', 'number': 2}]
    """
    return (dict(zip(dicts, x)) for x in itertools.product(*dicts.values()))

def read_sklearn_model(model_dir, model_prefix, single_model=False):
    # Read model from file
    models = {}
    # model_filename = model_path
    if single_model:
        models[0] = pickle.load(open(model_dir, 'rb'))
    else:
        for entry in os.scandir(model_dir):
            if entry.name.startswith(model_prefix) and entry.name.endswith('.pk'):
                indicator = int(entry.name[entry.name.rfind('_')+1:len(entry.name)-3])
                models[indicator] = pickle.load(open(entry, 'rb'))
    return models


def disable_output_cpx(instance_cpx):
    instance_cpx.set_log_stream(None)
    # instance_cpx.set_error_stream(None)
    instance_cpx.set_warning_stream(None)
    instance_cpx.set_results_stream(None)

def read_optval(filename):
    # Read true optval to get regret
    with open(filename, 'r') as file:
        results_str = file.read()
    objval_true = float(results_str.split(',')[-3])

    return objval_true
