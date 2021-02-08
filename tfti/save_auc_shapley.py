import os
import sys

# Dependency imports
from tensor2tensor.bin import t2t_trainer
from tensor2tensor.data_generators import text_encoder
from tensor2tensor.utils import decoding
from tensor2tensor.utils import registry
from tensor2tensor.utils import trainer_lib
from tensor2tensor.utils import usr_dir

# import tensorflow as tf

# for metrics
from sklearn import metrics
from sklearn.metrics import roc_auc_score

sys.path.append("../tfti")
import tfti

from itertools import combinations
import math
import bisect
import sys
from skpp import ProjectionPursuitRegressor
import numpy as np
import numpy.ma as ma
import random

sys.path.append("../tfti")
from tfti_batched_inference_multicell import *

sys.path.append("../shapley")
import shapley

import time
import sys

import scipy.io
from six.moves import xrange
import argparse


############## Functions ###################

def pseudo_batch(x, n):
    """Yields the value x n-times."""
    for _ in range(n):
        yield x
        
        
def filter_negatives(inputs, targets, keep_mask, n=10000):
    # mask each row in targets by keep mask
    # flip keep mask
    mask = np.invert(keep_mask.astype(bool))
    # tile to matrix size
    mask_matrix = np.tile(mask, (targets.shape[0], 1))

    # create masked targets
    masked_targets = np.ma.array(targets, mask = mask_matrix)
    x = np.sum(masked_targets, axis=1)

    # get row indices where masked sums are > 0
    filtered_indices = np.where(x>0)

    targets = targets[filtered_indices]
    inputs = inputs[filtered_indices]
    
    num_records = len(inputs)
    print(f"Using {num_records} samples for Shapley analysis")

    # only assess n sequences
    inputs = inputs[0:n]
    targets = targets[0:n]
    
    return inputs, targets
    

############## End Functions ###############




########## Command line arguments ##########
# get command line arguments
parser = argparse.ArgumentParser(description='Arguments for getting average AUC values on a Tensor2Tensor problem.')

parser.add_argument('output_file', metavar='output_file', type=str, nargs=1,
                   help='Output files to save values to.')

parser.add_argument('--model_checkpoint_path', metavar='model_checkpoint_path', type=str, nargs=1,
                    default="/data/akmorrow/tfti/t2t_train/6-128-25m/model.ckpt-164001",
                    help='Path to Tensor2Tensor model checkpoint')

parser.add_argument('--data_dir', metavar='data_dir', type=str, nargs=1,
                    default="/data/epitome/tmp/deepsea_train",
                   help='Directory containing DeepSEA .mat files (valid.mat and test.mat)')

parser.add_argument('--problem', metavar='problem', type=str, nargs=1,
                    default="genomics_binding_deepsea_multicell",
                   help='t2t problem string (default genomics_binding_deepsea_multicell)')

parser.add_argument('--model', metavar='model', type=str, nargs=1,
                    default="tfti_transformer",
                   help='t2t model string (default tfti_transformer)')

parser.add_argument('--hparams_set', metavar='hparams_set', type=str, nargs=1,
                    default="tfti_transformer_base",
                   help='t2t hparams set name(default tfti_transformer_base)')

parser.add_argument('--hparams', metavar='hparams', type=str, nargs=1,
                    default="multigpu=True",
                   help='t2t hparams string (default is \'\' )')

parser.add_argument('--hours', metavar='hours', type=int, nargs=1,
                    default=10,
                   help='Hours to run program for. The longer the program runs, the more samples it will take for calculating AUC')

parser.add_argument('--is_validation', dest='is_validation', action='store_true')
parser.add_argument('--is_test', dest='is_validation', action='store_false')

parser.add_argument('--subset_records', dest='subset_records',
                    action='store_true')



args = parser.parse_args()

# define the problem
problem_str=args.problem
model_str=args.model
hparams_set_str=args.hparams_set
hparams_str=args.hparams

# define file locations
tmp_dirname = args.data_dir
checkpoint_path = args.model_checkpoint_path

# define output filename
out_filename = args.output_file[0]
is_validation = args.is_validation
subset_records = args.subset_records

######## End Command line arguments #########


config = get_config(
    problem=problem_str,
    model=model_str,
    hparams_set=hparams_set_str,
    hparams=hparams_str,
    checkpoint_path=checkpoint_path,
)

cell_type="GM12878"

preprocess_batch_fn = get_preprocess_batch_fn(config, cell_type)
inference_fn = get_inference_fn(config)

# load in generator
tmp_dir = os.path.expanduser(tmp_dirname)

config = get_config(problem_str, model_str, hparams_set_str, hparams_str, checkpoint_path)
problem, model, hparams = get_problem_model_hparams(config)


# TODO do not hard code
all_marks = sorted(['CEBPB', 'CHD2', 'CTCF', 'DNase', 'EZH2', 'GABP', 'JunD', 'Max', 'Mxi1', 
                  'NRSF', 'Nrf1', 'Pol2', 'RFX5', 'Rad21', 'TAF1', 'TBP', 'USF2', 'c-Myc', 
                  'p300'])


########## Load data data #########
# Filter out non non-zero examples from test generator
keep_mask = np.array(get_keep_mask_for_marks(problem, all_marks, cell_type))

if (is_validation):
    filename = os.path.join(tmp_dir, "valid.mat")
    tmp = scipy.io.loadmat(filename)
    targets = tmp["validdata"]
    inputs = tmp["validxdata"]

else:
    filename = os.path.join(tmp_dir, "test.mat")
    tmp = scipy.io.loadmat(filename)
    targets = tmp["testdata"]
    inputs = tmp["testxdata"]
    
    
if (subset_records):
    inputs, targets = filter_negatives(inputs, targets, keep_mask, n=10000)
    
sequences = []
for i in xrange(inputs.shape[0]):
  sequences.append(problem.stringify(inputs[i].transpose([1, 0])))
inputs = sequences
num_records = len(inputs)

######################

marks_str = '\t'.join(all_marks)

# get all combs up to 2. This should take about 10 hours.
depth = 6

# Calculate number of iterations to run
iter_time = 1 # minute
num_iters = int(args.hours * (60/iter_time))
# Subset power_set by number of iterations. Do not sample single and pairwise iterations.
first_n = len(shapley.power_set(all_marks, 2)) # make sure to compute all single and pair wise
power_set = shapley.power_set(all_marks, depth=depth)
indices = random.sample(range(len(power_set[first_n:])), num_iters-first_n)
power_set = power_set[0:first_n] + [power_set[first_n:][i] for i in sorted(indices)]

iters = len(power_set)
this_iter = 0         
             
batch_size = 128

f= open(out_filename,"w+")
f.write(f"permutation\t{marks_str}\taverageAuROC\tmaskedAverageAuROC\n")

for set_ in power_set:
    if ((this_iter % 100) == 0):
        print(f"Computed {this_iter} out of {iters} in the power set")
             
    start = time.time()

    # select marks for this run
    selected_marks = [m for m in all_marks if m in set_]
    
    keep_mask = get_keep_mask_for_marks(problem, selected_marks, cell_type)
    
    # instantiate labels and predictions for this set
    labels_numpy = np.zeros((num_records, len(all_marks) ))
    predictions_numpy = np.zeros((num_records, len(all_marks) ))
    
    for i in range(0, num_records, batch_size):
        min_batch_size = min(batch_size, len(inputs)-i)
        if (min_batch_size > 0):
            batch_keep_mask = pseudo_batch(keep_mask, min_batch_size)

            batch = preprocess_batch_fn(
                inputs[i:i+min_batch_size],
                targets[i:i+min_batch_size],
                batch_keep_mask
            )
            response = inference_fn(batch)
            labels_numpy[i:i+min_batch_size] = response['labels'].reshape((min_batch_size, len(all_marks)))
            predictions_numpy[i:i+min_batch_size] = response['predictions'].reshape((min_batch_size, len(all_marks)))

    
    roc_aucs = []
    masked_roc_aucs = []
    for i in range(len(all_marks)):
        # Compute micro-average ROC area for all marks
        try:
            roc_auc = roc_auc_score(labels_numpy[:,i],predictions_numpy[:,i])
        except ValueError:
            roc_auc = np.NaN
        roc_aucs.append(roc_auc)
        if (all_marks[i] not in set_):
            masked_roc_aucs.append(roc_auc)
        
    roc_auc_str = '\t'.join(str(x) for x in roc_aucs)
    
    # filter out nans to compute auc
    roc_aucs = np.array(roc_aucs)
    masked_roc_aucs = np.array(masked_roc_aucs)
    
    average_roc = roc_aucs[np.logical_not(np.isnan(roc_aucs))].mean()
    masked_average_roc = masked_roc_aucs[np.logical_not(np.isnan(masked_roc_aucs))].mean()
    
    end = time.time()
    elapsed = end - start
    print("Set %s: Computed average auROC: %s in %s seconds" % (set_, average_roc, elapsed))
    
    # save values
    f.write("%s\t%s\t%s\t%s\n" % (selected_marks, roc_auc_str, average_roc, masked_average_roc))
    f.flush()
    
f.close()
