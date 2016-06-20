from __future__ import division

import argparse
import glob
import numpy
import scipy.optimize as opt
import os
import pickle
import re

"""
This script computes ensemble predictions based on multiple ASReader models.

In an optional first step it loads a dumped model and generates predictions from a validation and test dataset.
Subsequently it combines the predictions of several models using one of three methods
* AverageAll - the ensemble prediction is a mean of all the supplied single-model predictions
* pBest - sorts the candidate models by validation accuracy and selects the best proportion p of models to form the ensemble
* AddImprover - sorts the candidate models by validation accuracy and then tries adding them to the ensemble in that order
              keeping each model in the ensemble only if it improves its val. accuracy
Validation and test accuracies are printed for the ensemble and the best single model.

typical usage:
python fusion.py -pr "out_dir/best_predictions/*.y_hat_valid" -o out_dir/best_predictions/fusion.y_hat -t foo --fusion_method AverageAll
where the best_predictions directory should contain predictions selected by the copyBestPredictions.py script
"""


def accuracy_k_best(probas, mute=False, k_vals=[1,2,5]):
    """
    Gives the percentage of predictions that have the correct answer among the k most likely suggested answers for k=[1,2,5]
    :param probas: a list of numpy arrays, each containing a distribution of probabilities over candidates answers for one example
    :param mute: if True, stops the function from printing the accuracies into std out
    :param k_vals: values
    :return: an array of recall@k for k=1,2,5
    """
    recall_k = {}
    for k in k_vals:
        recall_k[k] = 0
        line = 0
        for row in probas:
            line += 1
            indices = numpy.argpartition(row, -k)[-k:]  # Gives indices of k highest probabilities
            recall_k[k] += (0 in indices)  # Uses the fact that correct answer is at index 0.
        recall_k[k] /= len(probas)
        if not mute:
            print 'recall@%d' % k, recall_k[k]
    return recall_k


def accuracy(probas):
    """
    Returns the proportion of predictions that assign the highest probability to the correct answer
    :param probas: a list of numpy arrays, each containing a distribution of probabilities over candidates answers for one example
    :return: accuracy
    """
    ncorrect = 0
    for row in probas:
        # We use the convention of having the ground truth answer at index 0
        ncorrect += (numpy.argmax(row) == 0)
    return ncorrect / len(probas)


def dump_model_predictions(model_filenames, input_data, suffix="y_hat", regenerate=False):
    """
    Loops through model files and uses the cbt_memory_pointer script to generate the y_hat predictions from each model
    :param model_filenames: list of filenames of saved models
    :param input_data: the dataset to which the models will be applied to generate predictions
    :param suffix: suffix of the generated prediction files (the rest of the filename is the same as the model filename
    :param regenerate: if true the model rewrites the prediction files even if they're already present
    :return: list of filenames of the generated predictions
    """
    prediction_filenames = []
    for model_file in model_filenames:
        y_hat_file = model_file + "." + suffix
        if not os.path.isfile(y_hat_file) or regenerate:
            load_model_command = 'python ' + args.blocks_nlp_path + 'as_reader.py --load_model ' + model_file + ' --y_hat_out_file ' + y_hat_file + ' --files_to_visualize ' + input_data + ' --no_html'
            if args.cnn:
                load_model_command += ' --dataset_type cnn'
            os.system(load_model_command)
        prediction_filenames.append(y_hat_file)
    return prediction_filenames


def adjust_length(pred_line, lineN, max_length):
    """
    Messy function that handles problems that arise if predictions for the same example have different lengths
    which may happen due to using a different batch size for each model. Normally it shouldn't be needed.
    :param pred_line:
    :param lineN:
    :param max_length:
    :return:
    """
    pred_line = numpy.trim_zeros(pred_line, trim='b')
    # The following takes care of lines that are shorter than the ones for previous files due to 0-trimming
    if lineN > len(max_length):
        maxLen = numpy.append(max_length, len(pred_line))
    while len(pred_line) < maxLen[lineN - 1]:
        pred_line = numpy.append(pred_line, 0)
    # print "Tail zero added to line "+str(lineN)+" of "+pred_file
    if len(pred_line) > maxLen[lineN - 1]:
        print '!!! Warning: Line ' + str(lineN) + ' is  longer than the corresponding lines of previous files.'
        maxLen[lineN - 1] = len(pred_line)
    return pred_line, max_length

def predictions_from_csv(fh, max_length):
    """
    Loads single model predictions from a csv file where lines may differ in length
    :param fh: file handle to the csv file
    :return: list of numpy arrays representing the predictions of individual examples
    """
    preds = list()
    lineN = 0
    for line in fh:
        lineN += 1
        pred_line = numpy.fromstring(line, sep=', ')
        if (args.trim_zeros):
            # If different batch sizes are used for the fused models, the prediction vectors need to be adjusted
            pred_line, max_length = adjust_length(pred_line, lineN, max_length)
        preds.append(pred_line)
    return preds, max_length


def load_all_predictions(prediction_filenames):
    # list of model predictions
    all_preds = []
    # list of corresponding accuracies
    model_accuracies=[]
    # the length of the longest prediction vector for each training example
    max_lengths = numpy.array([0])

    for pred_file in prediction_filenames:
        pred_fh = open(pred_file, 'r')
        # Predictions can be saved either in a csv or a pickle file
        if args.prediction_format == 'csv':
            preds, max_lengths =predictions_from_csv(pred_fh, max_lengths)
        else:
            preds = pickle.load(pred_fh)
        pred_fh.close()
        print "Results for " + pred_file
        acc = accuracy(preds)
        model_accuracies.append(acc)
        print 'Accuracy: ' + str(acc)
        all_preds.append(preds)

    return all_preds, model_accuracies

def fuse_predictions(prediction_filenames, weights=None):
    """
    reads the y_hat files and averages all the predictions
    :param prediction_filenames:
    :param weights: a list of weights given to the individual predictions within the fusion (defaults to equel weights)
    :return:
    """
    all_preds, model_accuracies = load_all_predictions(prediction_filenames)

    print
    print "Ensemble (equal weights): "
    ensemble_accuracy = accuracy(numpy.mean(all_preds, 0))
    print "Accuracy:\t"+str(ensemble_accuracy)
    # If weights were provided, calculate the prediction of a weighted ensemble
    averaged = numpy.average(all_preds, axis=0, weights=weights)
    if weights is not None:
        print "Weighted ensemble: "
        print "Accuracy:\t"+accuracy(averaged)
    return {'averaged': averaged, 'model_preds': all_preds, 'ensemble_acc': ensemble_accuracy,
            'model_accuracies': model_accuracies}



def greedy_add(prediction_filenames, greedy_iterations=1):
    """
    Builds up an ensemble by starting with the best validation model and then adding each model only if it improves
    the ensemble performance
    :param prediction_filenames: List of files containing candidate models' validation predictions
    :param greedy_iterations: int how many times each candidate model is considered for adding into the ensemble
    :return:
    """

    all_preds, model_accuracies = load_all_predictions(prediction_filenames)
    # Sort models by validation accuracy
    sorted_indices = numpy.argsort(model_accuracies)[::-1]
    # Indices of models included in the ensemble
    ensemble = numpy.array([], dtype='i')
    ensemble_accuracy = 0
    # List of predictions of models included in the ensemble
    member_predictions = []

    for _ in range(greedy_iterations):
        for i in sorted_indices:
            # Create a candidate ensemble and test whether it's better than the current ensemble
            if len(member_predictions) == 0:
                candidate_prediction = all_preds[i]
            else:
                candidate_member_predictions = member_predictions + [all_preds[i]]
                candidate_prediction = numpy.mean(candidate_member_predictions, 0)
            candidate_accuracy = accuracy(candidate_prediction)
            if candidate_accuracy > ensemble_accuracy:
                ensemble = numpy.hstack([ensemble, [i]])
                ensemble_accuracy = candidate_accuracy
                member_predictions.append(all_preds[i])

    print
    print 'Predictions included in the ensemble and their validation accuracies:'
    for i in ensemble:
        print str(model_accuracies[i]) + "\t" + prediction_filenames[i]
    best_single_valid_acc = model_accuracies[ensemble[0]]

    print
    print 'Ensemble accuracy: ' + str(ensemble_accuracy)
    ensemble_pred = numpy.mean(member_predictions, 0)
    return {'ensemble_prediction': ensemble_pred, 'ensemble_indices': ensemble,
            'ens_member_predictions': member_predictions, 'best_single_valid_acc': best_single_valid_acc,
            'ensemble_acc': ensemble_accuracy}


def p_best_models(prediction_filenames, p=0.7):
    """
    Sorts models by validation accuracy and forms the ensemble from the best ones. A proportion p of models is included.
    :param prediction_filenames:
    :param p: proportion of the provided models that is included in the ensemble
    :return:
    """
    all_preds, model_accuracies = load_all_predictions(prediction_filenames)

    # Sort models by validation accuracy
    sorted_indices = numpy.argsort(model_accuracies)[::-1]
    ensemble_size = int(p * len(sorted_indices))
    ensemble = sorted_indices[0:ensemble_size]

    # List of predictions of models included in the ensemble
    member_predictions = []
    for i in ensemble:
        member_predictions.append(all_preds[i])

    ensemble_pred = numpy.mean(member_predictions, 0)  # the ensemble prediction

    print
    print 'Predictions included in the ensemble and their validation accuracies:'
    for i in ensemble:
        print str(model_accuracies[i]) + "\t" + prediction_filenames[i]
    best_single_valid_acc = model_accuracies[ensemble[0]]
    print
    ensemble_accuracy = accuracy(ensemble_pred)
    print 'Ensemble accuracy: ' + str(ensemble_accuracy)
    ensemble_pred = numpy.mean(member_predictions, 0)
    # print 'Worse case: ' + str(accuracy_k_best(ensemble_pred)[1])
    return {'ensemble_prediction': ensemble_pred, 'ensemble_indices': ensemble,
            'ens_member_predictions': member_predictions, 'best_single_valid_acc': best_single_valid_acc,
            'ensemble_acc': ensemble_accuracy}


def optimize_weights(ensemble_indices, ens_member_predictions):
    """
	Optimizes the weights of models in the ensemble using the Constrained Optimization by Linear Approximation
    (COBYLA) method to maximize the validation accuracy
	:param ensemble_indices: list of indices of individual models that should be included in the optimized ensemble
	:param ens_member_predictions: list of prediction lists of the individual models
	:return: optimal weights, predictions of the optimal ensemble
    """

    def weight_accuracy(weights):
        # Objective function (negative accuracy) to be minimized
        averaged_pred = numpy.average(ens_member_predictions, axis=0, weights=weights)
        return -accuracy(averaged_pred)

    opt_result = opt.minimize(weight_accuracy, numpy.ones(len(ensemble_indices)) / len(ensemble_indices),
                              method='COBYLA',
                              constraints=({'type': 'ineq', 'fun': lambda x: 1 - sum(x)}))

    averaged_pred = numpy.average(ens_member_predictions, axis=0, weights=opt_result['x'])
    print 'Optimized ensemble accuracy: '
    print accuracy(averaged_pred)
    print 'Optimal weights: ' + str(opt_result['x'])
    return opt_result['x'], averaged_pred


def predictions2csv(predictions, fh):
    """
    Dump predictions in a csv format using filehandle fh
    """
    fh.write("\n".join(",".join(numpy.char.mod('%f', row)) for row in predictions))



"""
Parse command line arguments
"""

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                 description="Utility for fusing multiple classifier results.")

parser.add_argument('-m', '--models', nargs="+", default=None,
                    help='files containing models to fuse')

parser.add_argument('-mr', '--models_regexp', default=None,
                    help='regexp to match model files to fuse')

parser.add_argument('-d', '--input_data', default=None,
                    help='Input data to which we apply the models')

parser.add_argument('-t', '--test_data', default=None,
                    help='Test data for the ensemble')

parser.add_argument('-p', '--predictions', nargs="+", default=None,
                    help='files containing previously generated predictions')

parser.add_argument('-pr', '--prediction_regexp', default=None,
                    help='regexp to match prediction files to fuse')

parser.add_argument('-o', '--output', default=None,
                    help='file where fused predictions will be saved')

parser.add_argument('--blocks_nlp_path', default='~/dev/blocks-nlp/text_comprehension/',
                    help='absolute path of the directory containing as_reader.py ending with "(...)/blocks-nlp/text_comprehension/"')

parser.add_argument('--fusion_method', default='AddImprover', choices=['AddImprover', 'AverageAll', 'pBest'],
                    help='Choose the method of fusing models')

parser.add_argument('--greedy_iterations', type=int, default=1,
                    help='How many times the greedy algorithm iterates over the candidate models.')

parser.add_argument('--regenerate', action='store_true',
                    help='Force models to regenerate predictions even if they are already present in the directory')

parser.add_argument('--pred_file_suffix', default="",
                    help="Appends an additional suffix to prediction files - useful for regenerating predictions while keeping the old ones.")

parser.add_argument('--cnn', action='store_true', help='Indicates that datasets are in the CNN format.')

parser.add_argument('--optimize_weights', action='store_true',
                    help='Optimize weights of ensemble models to maximize validation accuracy.')

parser.add_argument('-f', '--prediction_format', default='csv', choices=['csv', 'pickle'],
                    help='format of the saved predictions (at the moment cannot generate csv from models)')

parser.add_argument('-es', '--ensemble_size', type=float, default=0.7,
                    help='proportion of models to be included in the ensemble (if relevant to the fusion method used)')

parser.add_argument('--trim_zeros', action='store_true',
                    help='Trims tail zeros of the predictions. Don\'t use for CBT.')

args = parser.parse_args()

# Filenames of dumped models to be used
to_fuse = []

# Add the model files from both arguments to a common array:
if args.models_regexp:
    to_fuse += glob.glob(args.models_regexp)
if args.models:
    to_fuse += args.models

print "Models to be fused:"
for model in enumerate(to_fuse):
    print model
print

# Save model predictions to disk and retain their paths
prediction_files = dump_model_predictions(to_fuse, args.input_data, 'y_hat_valid' + args.pred_file_suffix)

# Add previously generated prediction files if specified
if args.prediction_regexp:
    prediction_files += glob.glob(args.prediction_regexp)
if args.predictions:
    prediction_files += args.predictions

# Build the ensemble and generate the fused prediction:
if args.fusion_method == 'AddImprover':
    result = greedy_add(prediction_files, greedy_iterations=args.greedy_iterations)
    ensemble_indices = result['ensemble_indices']
    ens_member_predictions = result['ens_member_predictions']
    best_single_valid = result['best_single_valid_acc']

    if (args.optimize_weights):
        ens_weights, fused = optimize_weights(ensemble_indices, ens_member_predictions)
    else:
        fused = result['ensemble_prediction']
        ens_weights = None

elif args.fusion_method == 'pBest':
    result = p_best_models(prediction_files, args.ensemble_size)

    ens_member_predictions = result['ens_member_predictions']
    ensemble_indices = result['ensemble_indices']
    best_single_valid = result['best_single_valid_acc']

    if args.optimize_weights:
        ens_weights, fused = optimize_weights(ensemble_indices, ens_member_predictions)
    else:
        fused = result['ensemble_prediction']
        ens_weights = None

elif args.fusion_method == 'AverageAll':
    result = fuse_predictions(prediction_files)

    ens_member_predictions = result['model_preds']
    ensemble_indices = numpy.argsort(result['model_accuracies'])[::-1]
    best_single_valid = max(result['model_accuracies'])

    if args.optimize_weights:
        ens_weights, fused = optimize_weights(ensemble_indices, ens_member_predictions)
    else:
        fused = result['averaged']
        ens_weights = None

ensemble_valid = result['ensemble_acc']
print "Ensemble size: " + str(len(ensemble_indices))

# Optionally, save the fusion predictions
if args.output:
    output_fh = open(args.output, 'w')
    if args.prediction_format == 'csv':
            predictions2csv(fused,output_fh)
    else:
        pickle.dump(fused, output_fh)
    output_fh.close()
    print "Fused validation predictions saved to " + args.output

# Generate prediction files for ensemble models from test data
if args.test_data:
    print
    print '___________ Applying ensemble models to test data __________'
    # Create a list of filenames of dumped models included in the ensemble
    ensemble_models = []
    for i in ensemble_indices:
        ensemble_models += [re.sub('\.y_hat(_valid)?' + args.pred_file_suffix + '$', '', prediction_files[i])]
    # Use these models to generate test predictions
    prediction_files = dump_model_predictions(ensemble_models, args.test_data, 'y_hat_test' + args.pred_file_suffix)
    # Fuse these predictions
    result = fuse_predictions(prediction_files, ens_weights)
    ensemble_test_prediction = result['averaged'].squeeze()
    best_single_test = result['model_accuracies'][0]
    ensemble_test = result['ensemble_acc']
    # If required, dump the ensemble test prediction
    if args.output:
        output_fh = open(args.output + '_test', 'w')
        if args.prediction_format == 'csv':
            predictions2csv(ensemble_test_prediction,output_fh)
        else:
            pickle.dump(ensemble_test_prediction, output_fh)
        output_fh.close()
        print
        print "Fused test predictions saved to " + args.output + '_test'

print
print "Summary of results (model - valid. acc. - test acc.):"
print "Best single model:\t" + str(best_single_valid) + "\t" + str(best_single_test)
print args.fusion_method + " Ensemble:\t" + str(ensemble_valid) + "\t" + str(ensemble_test)
