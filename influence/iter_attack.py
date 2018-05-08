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
						step_size, project_fn, train_dataset):		
	"""
	Returns poisoned_X_train, a subset of model.data_sets.train (marked by indices_to_poison)
	that has been modified by a single gradient step.
	"""

	poisoned_X_train_subset = project_fn(
		train_dataset.x[indices_to_poison, :] - step_size * np.sign(grad_influence_wrt_input_val_subset) * 2.0 / 255.0)

	logger.info('-- max: %s, mean: %s, min: %s' % (
		np.max(grad_influence_wrt_input_val_subset),
		np.mean(grad_influence_wrt_input_val_subset),
		np.min(grad_influence_wrt_input_val_subset)))
	print(grad_influence_wrt_input_val_subset.shape)
	print(np.sign(grad_influence_wrt_input_val_subset).shape)
	print("boom: ", poisoned_X_train_subset.shape)
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
	top_model, full_model, top_graph, full_graph, project_fn, test_indices, test_description, 
	train_dataset, test_dataset, dataset_name,
	indices_to_poison=None,
	num_iter=10,
	step_size=1,
	save_iter=1,
	early_stop=None,
    beta = None,
	target_labels = None):
	"""
	targetted_attack: If True, it takes the label associated with the test point to be the target label to flip the model's prediction to.
					  If False, it takes the label associated with the test point as the true label, and just tried to get the model to predict any other one.
	
	"""

	#if early_stop is not None:
	#	assert len(test_indices) == 1, 'Early stopping only supported for attacks on a single test index.'

	if len(indices_to_poison) == 1:
		train_idx_str = indices_to_poison
	else:
		train_idx_str = len(indices_to_poison)

	top_model_name = top_model.model_name
	full_model_name = full_model.model_name
	
	# Remove everything but the poisoned train indices from the full model, to save time 
	labels_subset = train_dataset.labels[indices_to_poison]
	test_label = test_dataset.labels[test_indices[0]]

	logger.info('Test idx: {}, Indices to poison: {}, train label: {}, test label: {}'.format(test_indices, indices_to_poison, labels_subset, test_label))
	print("")
	

	train_f = np.load('output/%s_inception_features_new_train.npz' % dataset_name)
	inception_X_train = DataSet(train_f['inception_features_val'], train_f['labels'])
	test_f = np.load('output/%s_inception_features_new_test.npz' % dataset_name)
	inception_X_test = DataSet(test_f['inception_features_val'], test_f['labels'])

	validation = None
	
	with full_graph.as_default():
		test_pred = full_model.get_preds(test_dataset, test_indices)
		logger.info('Initial Test pred (full): %s' % test_pred)
	with top_graph.as_default():
		test_pred = top_model.get_preds(inception_X_test, test_indices)
		logger.info('Initial Test pred (top): %s' % test_pred)

	# inception_X_test_copy = DataSet(np.copy(inception_X_test.x), np.copy(inception_X_test.labels))
	# inception_X_test_copy.labels[test_indices] = target_labels
	
	inception_X_test_copy = None
	if target_labels == None:
		inception_X_test_copy =	inception_X_test
	else:
		inception_X_test_copy = DataSet(np.copy(inception_X_test.x), np.copy(inception_X_test.labels))
		inception_X_test_copy.labels[test_indices] = target_labels
	
	for attack_iter in range(num_iter):
		logger.info('*** Iter: %s' % attack_iter)
		logger.debug('Calculating perturbation...')
		# Use top model to quickly generate inverse HVP
		with top_graph.as_default():
			get_hvp(top_model, inception_X_test, inception_X_train, test_description, test_indices, True, target_labels)
			# get_hvp(top_model, inception_X_test_copy, inception_X_train, test_description, test_indices, True, target_labels)
		
		copyfile(
			'output/%s-test-%s.npz' % (top_model_name, test_description),
			'output/%s-test-%s.npz' % (full_model_name, test_description))

		# Use full model to get gradient wrt pixels
		with full_graph.as_default():
			for train_idx in indices_to_poison:
				# Calculate the pertubation
				grad_influence_wrt_input_val_subset = get_grad_of_influence_wrt_input(full_model, 
														test_indices, 
														None, # test_dataset
														[train_idx], train_dataset, 
														test_description,
														force_refresh=False)
				logger.info("Attach_iter {} perturbation shape: {}, perturbation: {}".format(attack_iter, grad_influence_wrt_input_val_subset.shape, grad_influence_wrt_input_val_subset))

				# If trying to increase loss, go in direction of gradient, otherwise go against direction of gradient
				if target_labels != None:
					grad_influence_wrt_input_val_subset*=-1.

				print(grad_influence_wrt_input_val_subset.shape)
				# New poisoned (raw - images) dataset after this iteration's pertubations
				poisoned_X_train_subset = poison_with_influence_proj_gradient_step(
					[train_idx],
					grad_influence_wrt_input_val_subset,
					step_size,
					project_fn, 
					train_dataset)
				
				print("train_idx: ", train_idx)
				print(poisoned_X_train_subset.shape)
				print(train_dataset.x[[train_idx], :].shape)

				# Update raw-image training dataset with poison
				train_dataset.x[[train_idx], :] = poisoned_X_train_subset

				with full_graph.as_default():
					# We update the cached inception features for the raw-image input with the poisoned version
					inception_X_train.x[train_idx, :] = generate_inception_features(full_model, poisoned_X_train_subset, labels_subset)


		with top_graph.as_default():
			# retrain top model on new inception features for poisoned dataset
			weights = top_model.retrain_and_get_weights(inception_X_train.x, inception_X_train.labels)
			weight_path = 'output/inception_weights_%s_attack_testidx-%s.npy' % (top_model_name, test_description)
			np.save(weight_path, weights)
		with full_graph.as_default():
			full_model.load_weights_from_disk(weight_path, do_save=False, do_check=False)

		# Print out attack effectiveness if it's not too expensive
		test_pred = np.zeros((len(test_indices), full_model.num_classes))
		if len(test_indices) < 100:
			#print("test_indices", test_indices)
			with full_graph.as_default():
				for test_idx in test_indices:
					pred = full_model.get_preds(test_dataset, [test_idx])
					logger.info('Test_idx: %s Test pred (full): %s'%(test_idx, pred))
			logger.info('---------------------')
			with top_graph.as_default():
				for counter, test_idx in enumerate(test_indices):
					pred = top_model.get_preds(inception_X_test, [test_idx])
					test_pred[counter, :] = pred
					logger.info('Test_idx: %s Test pred (top): %s'%(test_idx, pred))

			if early_stop is not None:
				done = True
				for counter, test_idx in enumerate(test_indices):
					if test_pred[counter, int(test_dataset.labels[[test_idx]])] > early_stop:
						done = False
						break
				if done:
					logger.info('Successfully attacked. Saving and breaking...')
					return train_dataset.x[indices_to_poison, :]

		# if (attack_iter+1) % save_iter == 0:
			# Exact same save code as above (Except for attach_iter)
			# np.savez('output/%s_attack_testidx-%s_trainidx-%s_stepsize-%s_proj_iter-%s' % (full_model.model_name, test_description, train_idx_str, step_size, attack_iter+1), 
			# 	poisoned_X_train_image=train_dataset[indices_to_poison], 
			# 	poisoned_X_train_inception_features=inception_X_train[indices_to_poison],
			# 	Y_train=labels_subset,
			# 	indices_to_poison=indices_to_poison,
			# 	attack_iter=attack_iter + 1,
			# 	test_pred=test_pred,
			# 	step_size=step_size)
	return False




