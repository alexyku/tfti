import os
import sys

# Dependency imports

from tensor2tensor.bin import t2t_trainer
from tensor2tensor.data_generators import text_encoder
from tensor2tensor.utils import decoding
from tensor2tensor.utils import registry
from tensor2tensor.utils import trainer_lib
from tensor2tensor.utils import usr_dir

import tensorflow as tf

# for metrics
from sklearn import metrics
from sklearn.metrics import roc_curve, auc

sys.path.append("../tfti")
import tfti
import tfti_infer



from itertools import combinations
import math
import bisect
import sys
from skpp import ProjectionPursuitRegressor
import numpy as np
import random

sys.path.append("../tfti")
from tfti_batched_inference import *

sys.path.append("../shapley")
import shapley

import time


def pseudo_batch(x, n):
    """Yields the value x n-times."""
    for _ in range(n):
        yield x
        
        
# define the problem
problem_str="genomics_binding_deepsea_gm12878"
model_str="tfti_transformer"
hparams_set_str="tfti_transformer_base"
hparams_str=""
checkpoint_path="/data/akmorrow/tfti/t2t_train/6-64-25/model.ckpt-210001"

config = get_config(
    problem="genomics_binding_deepsea_gm12878",
    model="tfti_transformer",
    hparams_set="tfti_transformer_base",
    hparams="",
    checkpoint_path="/data/akmorrow/tfti/t2t_train/6-64-25/model.ckpt-210001",

)

preprocess_batch_fn = get_preprocess_batch_fn(config)
inference_fn = get_inference_fn(config)

# load in validation generator
tmp_dir = os.path.expanduser("/data/epitome/tmp/")

config = tfti_infer.get_config(problem_str, model_str, hparams_set_str, hparams_str, checkpoint_path)
problem, model, hparams = tfti_infer.get_problem_model_hparams(config)
generator = problem.generator(tmp_dir, is_training=False)
generator_list = list(generator)


# reload(tfti_infer)
cell_type_1 = "GM12878"
cell_type_2 = "H1-hESC"

marks = tfti_infer.get_tfs(problem, cell_type_1, cell_type_2)
marks_str = '\t'.join(marks)

# get all combs up to 2. This should take about 10 hours.
depth = 2
power_set = shapley.power_set(marks, depth=depth)

batch_size = 64
inputs  = np.array(list(map(lambda x: x['inputs'], generator_list)))
targets = np.array(list(map(lambda x: x['targets'], generator_list)))

f= open(f"shapley_values_64_25_gm12878_depth_{depth}.txt","w+")
f.write(f"permutation\t{marks_str}\taverageAuROC\n")

for set_ in power_set:

    start = time.time()

    tf.logging.info("Computing average auROC for set %s" % set_)
    # select marks for this run
    selected_marks = [m for m in marks if m in set_]
    
    keep_mask = tfti_infer.get_keep_mask_for_marks(problem, selected_marks, cell_type_1)
    
    # instantiate labels and predictions for this set
    labels_numpy = np.zeros((len(generator_list), len(marks) ))
    predictions_numpy = np.zeros((len(generator_list), len(marks) ))
    
    for i in range(0, len(generator_list), batch_size):
        batch_keep_mask = pseudo_batch(keep_mask, batch_size)
        
        batch = preprocess_batch_fn(
            inputs[i:i+batch_size],
            targets[i:i+batch_size],
            batch_keep_mask

        )
        response = inference_fn(batch)
        labels_numpy[i:i+batch_size] = response['labels'].reshape((batch_size, len(marks)))
        predictions_numpy[i:i+batch_size] = response['predictions'].reshape((batch_size, len(marks)))
            
    end = time.time()
    elapsed = end - start
    print(f"Completed validation set in {elapsed} seconds")
    
    roc_aucs = []
    for i in range(len(marks)):
        # Compute micro-average ROC area for all marks
        fpr, tpr, _ = roc_curve(labels_numpy[:,i], predictions_numpy[:,i])
        roc_auc = auc(fpr, tpr)
        roc_aucs.append(roc_auc)
        
    roc_auc_str = '\t'.join(str(x) for x in roc_aucs)
    
    # filter out nans to compute auc
    roc_aucs = np.array(roc_aucs)
    average_roc = roc_aucs[np.logical_not(np.isnan(roc_aucs))].mean()
    tf.logging.info("Computed average auROC: %s" % (average_roc))
    
    # save values
    f.write("%s\t%s\t%s\n" % (selected_marks, roc_auc_str, average_roc))
    f.flush()
    
f.close()