import IPython
import numpy as np

import os
import time
from shutil import copyfile

from influence.inceptionModel import BinaryInceptionModel
from influence.binaryLogisticRegressionWithLBFGS import BinaryLogisticRegressionWithLBFGS
import influence.experiments
from influence.dataset import DataSet

import tensorflow as tf

from tensorflow.contrib.learn.python.learn.datasets import base
from influence.Progress import *

import logging
FORMAT = "[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
logger = logging.getLogger()
logging.basicConfig(format=FORMAT)
logger.setLevel(logging.INFO)
logging.debug("test")

def get_projection_to_box_around_orig_point(X_orig, box_radius_in_pixels=0.5):
  box_radius_in_float = box_radius_in_pixels * 2.0 / 255.0

  if X_orig is None:
    lower_bound = -1
    upper_bound = 1
  else:
    lower_bound = np.maximum(
      -np.ones_like(X_orig),
      X_orig - box_radius_in_float)
    upper_bound = np.minimum(
      np.ones_like(X_orig),
      X_orig + box_radius_in_float)

  # Automatically enforces -1, 1 as well
  def project_fn(X):
    return np.clip(X, lower_bound, upper_bound)

  return project_fn


def select_examples_to_attack(model, num_to_poison, influence_on_test_loss_val, step_size):

  # diffs = model.data_sets.train.x - np.clip(model.data_sets.train.x - step_size * np.sign(influence_on_test_loss_val) * 2.0 / 255.0, -1, 1) 
  # pred_diff = np.sum(diffs * influence_on_test_loss_val, axis = 1)    
  # This ignores the clipping, but it's faster    
  pred_diff = np.abs(influence_on_test_loss_val) 
  indices_to_poison = np.argsort(pred_diff)[-1:-num_to_poison-1:-1] # First index is the most effective
  return indices_to_poison

def poison_with_influence_proj_gradient_step(indices_to_poison, grad_influence_wrt_input_val_subset, 
                                             step_size, train_dataset):    
  """
  Returns poisoned_X_train, a subset of model.data_sets.train (marked by indices_to_poison)
  that has been modified by a single gradient step.
  """

  poisoned_X_train_subset = train_dataset.x[indices_to_poison, :] - step_size * np.sign(grad_influence_wrt_input_val_subset) * 2.0 / 255.0

  return poisoned_X_train_subset

def generate_inception_features(model, poisoned_X_train_subset, labels_subset, batch_size=None):
  poisoned_train = DataSet(poisoned_X_train_subset, labels_subset)    

  if batch_size == None:
    batch_size = len(labels_subset)

  print len(poisoned_X_train_subset), batch_size
  assert len(poisoned_X_train_subset) % batch_size == 0
  num_iter = int(len(poisoned_X_train_subset) / batch_size)

  poisoned_train.reset_batch()

  inception_features_val = []
  for i in xrange(num_iter):
    inception_features_val_temp = model.generate_inception_features(poisoned_train, batch_size)
    inception_features_val.append(inception_features_val_temp)

  print "Shape", np.concatenate(inception_features_val).shape
  return np.concatenate(inception_features_val)


def iterative_attack(
  top_model, full_model, top_graph, full_graph, 
  target_test_indices,
  test_description, # a string descirption of the test points
  train_dataset, test_dataset, dataset_name,
  indices_to_poison=None,
  num_iter=100, # number of iterations for the attack
  step_size=1, # step size used when applying the perturbation
  beta=None, # This argument is ignored, it's simply here to comply with the API call interface
  save_iter=None, # This argument is also ignored, though we can add a checkpointing mechanism if we wanted
  target_labels = None): # see below
  """
  targetted_attack: If True, it takes the label associated with the test point to be the target label to flip the model's prediction to.
            If False, it takes the label associated with the test point as the true label, and just tried to get the model to predict any other one.
  
  """

  # projection functionf or each training point to poison
  project_fn_list = [get_projection_to_box_around_orig_point(x, box_radius_in_pixels=0.5) for x in train_dataset.x[indices_to_poison]]

  top_model_name = top_model.model_name
  full_model_name = full_model.model_name
  
  # Just for printing information, these variables aren't used anywhere else
  poisoned_train_labels = train_dataset.labels[indices_to_poison]
  test_labels = test_dataset.labels[target_test_indices]
  logger.info('Test idx: {}, Indices to poison: {}, train label: {}, test labels: {}'.format(target_test_indices, indices_to_poison, poisoned_train_labels, test_labels))
  print("")
  
  train_features = np.load('output/%s_inception_features_new_train.npz' % dataset_name)
  inception_X_train = DataSet(train_features['inception_features_val'], train_features['labels'])
  test_features = np.load('output/%s_inception_features_new_test.npz' % dataset_name)
  inception_X_test = DataSet(test_features['inception_features_val'], test_features['labels'])

  with full_graph.as_default():
    test_pred = full_model.get_preds(test_dataset, target_test_indices)
    logger.info('Initial Test pred (full): %s' % test_pred)
  with top_graph.as_default():
    test_pred = top_model.get_preds(inception_X_test, target_test_indices)
    logger.info('Initial Test pred (top): %s' % test_pred)

  # As above, when non-targeted the label associated with the test points is the true label (to maximize loss for)
  # When targeted the label associated with the test points is the target label (to minimize loss for)
  inception_X_test_copy = None
  if target_labels == None:
    inception_X_test_copy = inception_X_test
  else:
    inception_X_test_copy = DataSet(np.copy(inception_X_test.x), np.copy(inception_X_test.labels))
    inception_X_test_copy.labels[target_test_indices] = target_labels
  
  for attack_iter in range(num_iter):
    logger.info('*** Iter: %s' % attack_iter)
    logger.debug('Calculating perturbation...')
    # Use top model to quickly generate inverse HVP
    with top_graph.as_default():
      get_hvp(top_model, inception_X_test, inception_X_train, test_description, target_test_indices, True, target_labels)
    
    copyfile(
      'output/%s-test-%s.npz' % (top_model_name, test_description),
      'output/%s-test-%s.npz' % (full_model_name, test_description))

    # Use full model to get gradient wrt pixels for each training point
    poisoned_X = np.zeros_like(train_dataset.x[indices_to_poison, :])
    with full_graph.as_default():
      # iterate over training points to poison
      for counter, train_idx in enumerate(indices_to_poison):
        # Calculate the pertubation
        grad_influence_wrt_input_val_subset = get_grad_of_influence_wrt_input(full_model, 
                                                                              target_test_indices, 
                                                                              None, # test_dataset
                                                                              [train_idx], train_dataset, 
                                                                              test_description,
                                                                              force_refresh=False)
        logger.info("Attach_iter {} perturbation shape: {}, perturbation: {}".format(attack_iter, grad_influence_wrt_input_val_subset.shape, grad_influence_wrt_input_val_subset))

        # If trying to increase loss, go in direction of gradient, otherwise go against direction of gradient
        if target_labels != None:
          grad_influence_wrt_input_val_subset*=-1.

        print(grad_influence_wrt_input_val_subset.shape)
        # The poisoned images after this iteration's pertubations
        poisoned_X_train_subset = poison_with_influence_proj_gradient_step(
          [train_idx],
          grad_influence_wrt_input_val_subset,
          step_size, 
          train_dataset)
        
        poisoned_X[counter, :] = poisoned_X_train_subset

      # project after each iteration to keep in allowed image bounds
      for i in range(len(indices_to_poison)):
          poisoned_X[i, :] = project_fn_list[i](poisoned_X[i])

      # Update raw-image training dataset with poison
      train_dataset.x[indices_to_poison, :] = poisoned_X

      # We update the cached inception features for the raw-image input with the poisoned version
      inception_X_train.x[indices_to_poison, :] = generate_inception_features(full_model, 
                                                                              train_dataset.x[indices_to_poison, :], 
                                                                              train_dataset.labels[indices_to_poison])

    # finished with perturbing all training points for this iteration, now retraining on dataset after this iteration's poison
    with top_graph.as_default():
      # retrain top model on new inception features for poisoned dataset
      weights = top_model.retrain_and_get_weights(inception_X_train.x, inception_X_train.labels)
      weight_path = 'output/inception_weights_%s_attack_testidx-%s.npy' % (top_model_name, test_description)
      np.save(weight_path, weights)
    with full_graph.as_default():
      full_model.load_weights_from_disk(weight_path, do_save=False, do_check=False)

    # Print out attack effectiveness if number of target test images is less than 100 (so it's not too expensive)
    target_test_pred = np.zeros((len(target_test_indices), full_model.num_classes))
    if len(target_test_indices) < 100:
      #print("target_test_indices", target_test_indices)
      with full_graph.as_default():
        for test_idx in target_test_indices:
          pred = full_model.get_preds(test_dataset, [test_idx])
          logger.info('Test_idx: %s Test pred (full): %s'%(test_idx, pred))
      logger.info('---------------------')
      with top_graph.as_default():
        for counter, test_idx in enumerate(target_test_indices):
          pred = top_model.get_preds(inception_X_test, [test_idx])
          target_test_pred[counter, :] = pred
          logger.info('Test_idx: %s Test pred (top): %s'%(test_idx, pred))


    # # Trying to do early stopping
    # # As mentioned above, test_dataset.labels for indices_to_poison are the true labels when untargeted, and the target labels when targeted
    # # If untargeted, we check that the label with the highest score (from target_test_pred) is NOT the true label
    # # If targeted, we check that the label with the highest score (from target_test_pred) IS the target label
    # label_with_highest_score = np.argmax(target_test_pred, axis=1)
    # equal_array = np.equal(label_with_highest_score, test_dataset.labels[target_test_indices])
    # if target_labels == None: # Untargeted
    #   done = not np.any(equal_array) # Done if none of the labels with the highest score are the true labels
    # else:
    #   done = np.all(equal_array) # Done if all of the labels with the highest score are the target labels
    # if done:
    #   logger.info('Successfully attacked. Saving and breaking...')
    #   return train_dataset.x[indices_to_poison, :]

  # return poisons even if attack failed
  return train_dataset.x[indices_to_poison, :]

