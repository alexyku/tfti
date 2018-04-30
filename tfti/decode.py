import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import collections

from tensor2tensor import models
from tensor2tensor import problems
from tensor2tensor.layers import common_layers
from tensor2tensor.utils import trainer_lib
from tensor2tensor.utils import t2t_model
from tensor2tensor.utils import registry
from tensor2tensor.utils import metrics

sys.path.append("./tfti")
import tfti

import scipy

class Decode:
    def __init__(self, ckpt_path, problem_name = "genomics_binding_deepsea_gm12878",
                model_name   = "tfti_transformer",
                hparams_set  = "tfti_transformer_base",
                hparams_overrides_str=""):
        data_dir  = "unused"  # will error out if `None` or empty-string
        mode = tf.estimator.ModeKeys.PREDICT  # set dropout to 0.0
        
        self.ckpt_path = ckpt_path
        self.model_name = model_name
        self.hparams  = trainer_lib.create_hparams(hparams_set, hparams_overrides_str, data_dir, problem_name)
        self.model    = registry.model(model_name)(self.hparams, mode)
        self.problem  = registry.problem(problem_name)
        self.encoders = self.problem.get_feature_encoders()

    def get_raw_data_generator(self, validation_file):
        """Yields raw inputs and targets.

        Yields:
            Tuples containing:
                inputs: NACTG strings of length 1000.
                targets: An binary label array of length 919.
        """
        # load in DEV data
        tmp = scipy.io.loadmat(validation_file)
        inputs = tmp["validxdata"]
        targets = tmp["validdata"]
        
#         tf.logging.info(targets[0,:])
        for i in range(inputs.shape[0]):
            yield self.problem.stringify(inputs[i].transpose([1, 0])), targets[i]


    def get_processed_data_generator_fn(self, raw_data_generator, keep_mask, targets_gather_indices):
        # Reshape to rank 3 arrays/tensors.
        keep_mask = keep_mask.reshape([-1, 1, 1])
        
        def get_processed_data_generator():
            for raw_inputs, raw_targets in raw_data_generator:
                preprocessed_inputs = np.array(self.encoders["inputs"].encode(raw_inputs), dtype=np.int64)
                preprocessed_targets = raw_targets[targets_gather_indices]
                # Reshape to rank 3 arrays/tensors.
                preprocessed_inputs = preprocessed_inputs.reshape([-1, 1, 1])
                preprocessed_targets = preprocessed_targets.reshape([-1, 1, 1])
                yield preprocessed_inputs, preprocessed_targets, keep_mask
        return get_processed_data_generator


    def get_keep_mask_for_marks(self, mark_names, tf_names_indices):
        """
        Creates a keep mask, keeping the mark_names specified
        :param mark_names list of "Dnase, "CTCF", etc.
        :tf_names_indices ordered list of model label names
        """
        all_marks = list(map(lambda x: x[1].split("|")[1], tf_names_indices))
        return 1 * np.isin(all_marks, mark_names)


    def get_latents_and_metrics_weights(self, targets, keep_mask):
        """Creates latents and weights."""
        metrics_mask = tf.to_float(tf.logical_not(keep_mask))
        float_keep_mask = tf.to_float(keep_mask)
        latents = tf.to_int32(
            float_keep_mask * tf.to_float(targets)
            + (1.0 - float_keep_mask) * self.problem.unk_id)
        return latents, metrics_mask
    
    
    def get_tfs(self, dev_cell_type_1, dev_cell_type_2):
            tf_names_indices = self.problem.get_overlapping_indices_for_cell_type(dev_cell_type_1, dev_cell_type_2)[1]
            return list(map(lambda x: x[1].split('|')[1], tf_names_indices))


    def infer(self, validation_file, validation_marks, dev_cell_type_1, dev_cell_type_2):
        """Runs inference on a single example."""

        tf.reset_default_graph()
        
        # Shapes for preprocessed inputs/targets/latents.
        preprocessed_input_sequence_length = int(np.ceil(self.problem.input_sequence_length / self.problem.chunk_size))
        preprocessed_num_binary_predictions = len(self.problem.targets_gather_indices(dev_cell_type_1, dev_cell_type_2))
        targets_gather_indices = self.problem.targets_gather_indices(dev_cell_type_1, dev_cell_type_2) # testing on new cell type
        tf_names_indices = self.problem.get_overlapping_indices_for_cell_type(dev_cell_type_1, dev_cell_type_2)[1]

        # get mask based on chosen marks
        # define which marks you would like to keep in the mask
        keep_mask = self.get_keep_mask_for_marks(validation_marks, tf_names_indices)

        # Create dataset from generator.
        raw_data_generator = self.get_raw_data_generator(validation_file)
        processed_data_generator_fn = self.get_processed_data_generator_fn(raw_data_generator, keep_mask, targets_gather_indices)
        ds = tf.data.Dataset.from_generator(
            processed_data_generator_fn,
            output_types=(tf.int64, tf.int64, tf.bool),
            output_shapes=(
                [preprocessed_input_sequence_length, 1, 1],
                [preprocessed_num_binary_predictions, 1, 1],
                [preprocessed_num_binary_predictions, 1, 1],
            )
        )

        ds = ds.repeat(1)  # Single evaluation epoch.
        ds = ds.batch(1)

        # Create one-shot-iterator.
        next_item = ds.make_one_shot_iterator().get_next()
        preprocessed_inputs, preprocessed_targets, latents_keep_mask = next_item


        # Create the latents from the targets and mask.
        latents, metrics_mask = self.get_latents_and_metrics_weights(preprocessed_targets, latents_keep_mask)

        # Preprocess examples.
        features = {
            "inputs": preprocessed_inputs,
            "targets": preprocessed_targets,
            "target_space_id": 0,  # generic id space
            "latents": latents,
            "latent_keep_mask": keep_mask
        }

        # Features to logits.
        with tf.variable_scope(self.model_name):
            logits, _ = self.model.model_fn(features) # broken here: variance between feature estimations is too small (e-15)
            predictions = tf.nn.sigmoid(logits)
            labels = features["targets"]


        # Add an op to initialize the variables.
        init_op = tf.global_variables_initializer()

        # Add ops to save and restore all the variables.
        variables_to_restore = [
            v for v in tf.contrib.slim.get_variables_to_restore()
            if "global_step" not in v.name
        ]
        saver = tf.train.Saver(variables_to_restore)

        # Initialize and restore variables.
        sess = tf.InteractiveSession()
        sess.run(init_op)
        saver.restore(sess, self.ckpt_path)

        predictions_and_labels = []

        i=0

        try:
            while True:
                if (i % 1000 == 0):
                     print(f"Computed predictions for : {i} points...")

                i += 1
                fetch = (predictions, labels)

                fetch_numpy =  sess.run(fetch)
                predictions_and_labels.append(fetch_numpy)

        except tf.errors.OutOfRangeError:
            print(f"Computed predictions for : {len(predictions_and_labels)} points")

        predictions_and_labels = [(x.squeeze(), y.squeeze())
                          for (x, y) in predictions_and_labels]

        predictions_numpy = np.array(predictions_and_labels)[:, 0, :]
        labels_numpy = np.array(predictions_and_labels)[:, 1, :]

        return (predictions_numpy, labels_numpy)
            
