from __future__ import division

import argparse
import glob
import numpy as np
import scipy.optimize as opt
import os
import pickle
import re
import shutil

"""
Scans a folder with .prediction files in csv format containing multiple predictions per model (e.g. one from each epoch)
The script chooses the best validation prediction for each model and copies it into a new folder together with the corresponding
test prediction with .y_hat_valid and .y_hat_test suffixes. 'fusion.py -pr "dir/*.y_hat_valid" -t foo' can then be applied to this folder to generate fusions.
"""


def predictionAccuracy(file):
	# Calculates the accuracy of a csv prediction file with ground truth on first position
	ncorrect=0
	nlines=0
	with open(file) as fh:
		for line in fh:
			prediction = np.fromstring(line, sep=', ')
			ncorrect += (prediction.argmax() == 0)
			nlines+=1
	return ncorrect / (1.0*nlines)


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                 description="Utility for fusing multiple classifier results.")


parser.add_argument('-vp', '--validation_prefix', default='validation_cleaned.txt.',
                    help='regexp to match validation predictions')

parser.add_argument('-tp', '--test_prefix', default='training_cleaned.txt.',
                    help='regexp to match prediction files to fuse')

parser.add_argument('--alt_test_prefix', default='test_cleaned.txt.',
					help='''test prefix to be tried if a file with the above one is unavailable
						 (test predictions were initially misnamed due to a bug resulting in two name conventions in one direcotry)''')

parser.add_argument('-s', '--suffix', default='.prediction',
                    help='suffix of the prediction files in the source folder')
					
parser.add_argument('-i', '--input_dir', default='.',
                    help='input directory containing all prediction files')
					
parser.add_argument('-o', '--output_dir', default='./predsForFusion',
                    help='output directory where to copy the best predictions')
                    
parser.add_argument('--output_prefix', default='',
                    help='prefixes for the copied files in order to allow fusing predictions from models with identical hyperparameters')


args = parser.parse_args()
to_fuse = []

# List of files containing validation predictions:
if args.input_dir and not re.search('\/$',args.input_dir):
    args.input_dir+="/"
validation_files = glob.glob(args.input_dir + args.validation_prefix + '*' + args.suffix)

# Create a dict of best validation predictions for each model and their accuracies:
bestPredictions=dict()
print "Validation files:"
for valFile in validation_files:
	print valFile
    # Prediction filename stemming to get a root which is common to all prediction files from the given model
	params_match=re.search(args.validation_prefix+'(.*)\.e\d+i\d+'+args.suffix,valFile)
	if params_match:
		param_string = params_match.group(1)
		print 'Parameter key: '+ param_string
		valAccuracy = predictionAccuracy(valFile)
		print 'Validation accuracy: '+str(valAccuracy)
        # If this is the best prediction so far, update the value in the bestPredictions dict
		if param_string in bestPredictions:
			if bestPredictions[param_string]['accuracy']< valAccuracy:
				bestPredictions[param_string]['accuracy'] = valAccuracy
				bestPredictions[param_string]['file'] = valFile
		else:
			bestPredictions[param_string]=dict()
			bestPredictions[param_string]['accuracy'] = valAccuracy
			bestPredictions[param_string]['file'] = valFile
	else:
		print '!!! Failed to parse filename as a prediction !!!'
	
print
print "Best predictions to be copied:"

# Model with the best validation accuracy and its stats
bestValModel={'validation':0}

if not os.path.exists(args.output_dir):
	os.makedirs(args.output_dir)

# Now we copy each best validation prediction together with the corresponding test prediction to an output dir. 
for params in bestPredictions:
	print params
	print 'Validation accuracy: '+str(bestPredictions[params]['accuracy'])
	shutil.copyfile(bestPredictions[params]['file'],args.output_dir+'/'+args.output_prefix+params+'.y_hat_valid')
	try:
		src_test_file=re.sub(args.validation_prefix,args.test_prefix,bestPredictions[params]['file'])
		print 'Source test prediction: '+src_test_file
		test_accuracy=predictionAccuracy(src_test_file)
	except IOError:
        # If there are two filename formats for the predictions, we can use an alternative prefix if the primary one fails
		src_test_file=re.sub(args.validation_prefix,args.alt_test_prefix,bestPredictions[params]['file'])
		print 'Source test prediction: '+src_test_file
		test_accuracy=predictionAccuracy(src_test_file)
	print 'Test accuracy: '+str(test_accuracy)
	shutil.copyfile(src_test_file,args.output_dir+'/'+args.output_prefix+params+'.y_hat_test')
	# Keep track of the best-accuracy single model
	if bestPredictions[params]['accuracy']>bestValModel['validation']:
		bestValModel['validation']=bestPredictions[params]['accuracy']
		bestValModel['test']=test_accuracy
		bestValModel['params']=params
	print
		
print		
print 'Best validation model:'
print bestValModel['params']
print 'Validation: '+str(bestValModel['validation'])
print 'Test: '+str(bestValModel['test'])
