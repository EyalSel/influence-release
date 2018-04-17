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


def select_examples_to_attack(model, num_to_poison, grad_influence_wrt_input_val, step_size):

	# diffs = model.data_sets.train.x - np.clip(model.data_sets.train.x - step_size * np.sign(grad_influence_wrt_input_val) * 2.0 / 255.0, -1, 1) 
	# pred_diff = np.sum(diffs * grad_influence_wrt_input_val, axis = 1)    
	# This ignores the clipping, but it's faster    
	pred_diff = np.sum(np.abs(grad_influence_wrt_input_val), axis = 1)
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

	print('-- max: %s, mean: %s, min: %s' % (
		np.max(grad_influence_wrt_input_val_subset),
		np.mean(grad_influence_wrt_input_val_subset),
		np.min(grad_influence_wrt_input_val_subset)))

	return poisoned_X_train_subset

def generate_inception_features(model, poisoned_X_train_subset, labels_subset, batch_size=None):
	poisoned_train = DataSet(poisoned_X_train_subset, labels_subset)    

	if batch_size == None:
		batch_size = len(labels_subset)

	assert len(poisoned_X_train_subset) % batch_size == 0
	num_iter = int(len(poisoned_X_train_subset) / batch_size)

	poisoned_data_sets.train.reset_batch()

	inception_features_val = []
	for i in xrange(num_iter):
		inception_features_val_temp = model.generate_inception_features(poisoned_train, batch_size)
		inception_features_val.append(inception_features_val_temp)

	return np.concatenate(inception_features_val)


def iterative_attack(
	top_model, full_model, top_graph, full_graph, project_fn, test_indices, test_description, 
	train_dataset, test_dataset, 
	indices_to_poison=None,
	num_iter=10,
	step_size=1,
	save_iter=1,
	early_stop=None):   

	if early_stop is not None:
		assert len(test_indices) == 1, 'Early stopping only supported for attacks on a single test index.'

	if len(indices_to_poison) == 1:
		train_idx_str = indices_to_poison
	else:
		train_idx_str = len(indices_to_poison)

	top_model_name = top_model.model_name
	full_model_name = full_model.model_name

	print('Test idx: %s' % test_indices)
	print('Indices to poison: %s' % indices_to_poison)

	# Remove everything but the poisoned train indices from the full model, to save time 

	labels_subset = train_dataset.labels[indices_to_poison]

	train_f = np.load('output/%s_inception_features_new_train.npz' % dataset_name)
	inception_X_train = DataSet(train_f['inception_features_val'], train_f['labels'])
	test_f = np.load('output/%s_inception_features_new_test.npz' % dataset_name)
	inception_X_test = DataSet(test_f['inception_features_val'], test_f['labels'])

	validation = None

	for attack_iter in range(num_iter):
		print('*** Iter: %s' % attack_iter)
		
		print('Calculating grad...')

		# Use top model to quickly generate inverse HVP
		with top_graph.as_default():
			get_hvp(top_model, inception_X_test, inception_X_train, test_description, True)
		copyfile(
			'output/%s-test-%s.npz' % (top_model_name, test_description),
			'output/%s-test-%s.npz' % (full_model_name, test_description))

		# Use full model to get gradient wrt pixels
		with full_graph.as_default():
			grad_influence_wrt_input_val_subset =  get_grad_of_influence_wrt_input(full_model, 
													test_data, indices_to_poison, train_data, 
													test_description,
													force_refresh=True)
   
			poisoned_X_train_subset = poison_with_influence_proj_gradient_step(
				indices_to_poison,
				grad_influence_wrt_input_val_subset,
				step_size,
				project_fn, 
				train_dataset)

		with full_graph.as_default():
			inception_X_train[indices_to_poison] = generate_inception_features(full_model, poisoned_X_train_subset, labels_subset)


		with top_graph.as_default():
			weights = top_model.retrain_and_get_weights(inception_X_train, train_dataset.labels)
			weight_path = 'output/inception_weights_%s_attack_testidx-%s.npy' % (top_model_name, test_description)
			np.save(weight_path, weights)
		with full_graph.as_default():            
			full_model.load_weights_from_disk(weight_path, do_save=False, do_check=False)

		# Print out attack effectiveness if it's not too expensive
		test_pred = None
		if len(test_indices) < 100:
			with full_graph.as_default():
				test_pred = full_model.get_preds(test_dataset, test_indices)
				print('Test pred (full): %s' % test_pred)
			with top_graph.as_default():
				test_pred = top_model.get_preds(test_dataset, test_indices)
				print('Test pred (top): %s' % test_pred)

			if ((early_stop is not None) and (len(test_indices) == 1)):
				if test_pred[0, int(test_dataset.labels[test_indices])] < early_stop:
					print('Successfully attacked. Saving and breaking...')
					np.savez('output/%s_attack_testidx-%s_trainidx-%s_stepsize-%s_proj_final' % (full_model.model_name, test_description, train_idx_str, step_size), 
						poisoned_X_train_image=train_dataset[indices_to_poison], 
						poisoned_X_train_inception_features=inception_X_train[indices_to_poison],
						Y_train=labels_subset,
						indices_to_poison=indices_to_poison,
						attack_iter=attack_iter + 1,
						test_pred=test_pred,
						step_size=step_size)            
					return True

		if (attack_iter+1) % save_iter == 0:
			np.savez('output/%s_attack_testidx-%s_trainidx-%s_stepsize-%s_proj_iter-%s' % (full_model.model_name, test_description, train_idx_str, step_size, attack_iter+1), 
				poisoned_X_train_image=train_dataset[indices_to_poison], 
				poisoned_X_train_inception_features=inception_X_train[indices_to_poison],
				Y_train=labels_subset,
				indices_to_poison=indices_to_poison,
				attack_iter=attack_iter + 1,
				test_pred=test_pred,
				step_size=step_size)
	return False




