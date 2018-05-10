import os
import sys

# Dependency imports


# TODO params
# valid or test
# sample or dont
# directories
# problem

from tensor2tensor.bin import t2t_trainer
from tensor2tensor.data_generators import text_encoder
from tensor2tensor.utils import decoding
from tensor2tensor.utils import registry
from tensor2tensor.utils import trainer_lib
from tensor2tensor.utils import usr_dir

import tensorflow as tf

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
from tfti_batched_inference import *

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

############## End Functions ###############

# get command line arguments
parser = argparse.ArgumentParser(description='Arguments for getting average AUC values on a Tensor2Tensor problem.')

parser.add_argument('model_checkpoint_path', metavar='model checkpoint path', type=str, nargs='+',
                   help='Path to Tensor2Tensor model checkpoint')


args = parser.parse_args()

        

        

### files
tmp_dirname = "/data/epitome/tmp/"
checkpoint_path = "/data/akmorrow/tfti/t2t_train/6-64-25/model.ckpt-210001"

        
        
# define the problem
problem_str="genomics_binding_deepsea_gm12878"
model_str="tfti_transformer"
hparams_set_str="tfti_transformer_base"
hparams_str=""

config = get_config(
    problem=problem_str,
    model=model_str,
    hparams_set=hparams_set_str,
    hparams=hparams_str,
    checkpoint_path=checkpoint_path,
)

preprocess_batch_fn = get_preprocess_batch_fn(config)
inference_fn = get_inference_fn(config)

# load in validation generator
tmp_dir = os.path.expanduser(tmp_dirname)


config = get_config(problem_str, model_str, hparams_set_str, hparams_str, checkpoint_path)
problem, model, hparams = get_problem_model_hparams(config)

cell_type_1 = "GM12878"
cell_type_2 = "H1-hESC"

# TODO do not hard code
all_marks = ['GM12878|GABP|None', 'GM12878|Egr-1|None', 'GM12878|NRSF|None',
       'GM12878|DNase|None', 'GM12878|CTCF|None', 'GM12878|EZH2|None',
       'GM12878|ATF3|None', 'GM12878|p300|None', 'GM12878|Pol2-4H8|None',
       'GM12878|SIX5|None', 'GM12878|c-Myc|None', 'GM12878|SRF|None',
       'GM12878|TAF1|None', 'GM12878|YY1|None', 'GM12878|USF-1|None',
       'GM12878|CHD1|None', 'GM12878|CHD2|None', 'GM12878|JunD|None',
       'GM12878|Max|None', 'GM12878|Nrf1|None', 'GM12878|RFX5|None',
       'GM12878|TBP|None', 'GM12878|ATF2|None', 'GM12878|BCL11A|None',
       'GM12878|CEBPB|None', 'GM12878|Pol2|None', 'GM12878|Rad21|None',
       'GM12878|RXRA|None', 'GM12878|SP1|None', 'GM12878|TCF12|None',
       'GM12878|BRCA1|None', 'GM12878|Mxi1|None', 'GM12878|SIN3A|None',
       'GM12878|USF2|None', 'GM12878|Znf143|None']


all_marks = list(map(lambda x: x.split('|')[1], all_marks))


###### Get test data #####
# Filter out non non-zero examples from test generator
keep_mask = np.array(get_keep_mask_for_marks(problem, all_marks, cell_type_1))

filename = os.path.join(tmp_dir, "deepsea_train/test.mat")
tmp = scipy.io.loadmat(filename)
targets = tmp["testdata"]
inputs = tmp["testxdata"]

# mask each row in targets by keep mask
# flip keep mask
mask = np.invert(keep_mask.astype(bool))
# tile to matrix size
mask_matrix = np.tile(mask, (targets.shape[0], 1))

# create masked targets
# masked_targets = np.ma.array(targets, mask = mask_matrix)
# x = np.sum(masked_targets, axis=1)

# get row indices where masked sums are > 0
# filtered_indices = np.where(x>0)

# targets = targets[filtered_indices]
# inputs = inputs[filtered_indices]
num_records = len(inputs)
print(f"Using {num_records} samples for Shapley analysis")

sequences = []
for i in xrange(inputs.shape[0]):
  sequences.append(problem.stringify(inputs[i].transpose([1, 0])))
    
# only assess 10000 sequences
# inputs = sequences[0:10000]
# targets = targets[0:10000]

######################


marks_str = '\t'.join(all_marks)

# get all combs up to 2. This should take about 10 hours.
depth = 5
power_set = shapley.power_set(all_marks, depth=depth)
iters = len(power_set)
this_iter = 0         
             
batch_size = 128

# define output filename
out_filename = "TESTESTauc_values_{problem_str}_depth_{depth}_tfCount_{len(all_marks)}.txt"

f= open(out_filename,"w+")
f.write(f"permutation\t{marks_str}\taverageAuROC\tmaskedAverageAuROC\n")

for set_ in power_set:
    if ((this_iter % 100) == 0):
        print(f"Computed {this_iter} out of {iters} in the power set")
             
    start = time.time()

    # select marks for this run
    selected_marks = [m for m in all_marks if m in set_]
    
    keep_mask = get_keep_mask_for_marks(problem, selected_marks, cell_type_1)
    
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
    for i in list(zip(all_marks, roc_aucs)):
        print(i)
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