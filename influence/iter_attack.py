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

def poison_with_influence_proj_gradient_step(model, indices_to_poison, grad_influence_wrt_input_val_subset, step_size, project_fn):    
    """
    Returns poisoned_X_train, a subset of model.data_sets.train (marked by indices_to_poison)
    that has been modified by a single gradient step.
    """
    poisoned_X_train_subset = project_fn(
        model.data_sets.train.x[indices_to_poison, :] - step_size * np.sign(grad_influence_wrt_input_val_subset) * 2.0 / 255.0)

    print('-- max: %s, mean: %s, min: %s' % (
        np.max(grad_influence_wrt_input_val_subset),
        np.mean(grad_influence_wrt_input_val_subset),
        np.min(grad_influence_wrt_input_val_subset)))

    return poisoned_X_train_subset


def iterative_attack(
	top_model, full_model, top_graph, full_graph, project_fn, test_indices, test_description, 
	train_dataset, test_dataset, 
	indices_to_poison=None,
    num_iter=10,
    step_size=1,
    save_iter=1,
    loss_type='normal_loss',
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
    poisoned_dataset = DataSet(train_dataset[indices_to_poison, :], )

    full_model.update_train_x_y(
        full_model.data_sets.train.x[indices_to_poison, :],
        full_model.data_sets.train.labels[indices_to_poison])
    eff_indices_to_poison = np.arange(len(indices_to_poison))
    labels_subset = full_model.data_sets.train.labels[eff_indices_to_poison]

    for attack_iter in range(num_iter):
        print('*** Iter: %s' % attack_iter)
        
        print('Calculating grad...')

        # Use top model to quickly generate inverse HVP
        with top_graph.as_default():
        	predicted_loss_diffs = get_influence_on_test_loss(top_model, 
		                               test_dataset, indices_to_poison, train_data, 
		                               test_description,
		                               force_refresh=True)
        copyfile(
            'output/%s-cg-%s-test-%s.npz' % (top_model_name, loss_type, test_description),
            'output/%s-cg-%s-test-%s.npz' % (full_model_name, loss_type, test_description))

        # Use full model to get gradient wrt pixels
        with full_graph.as_default():
        	grad_influence_wrt_input_val_subset =  get_grad_of_influence_wrt_input(full_model, 
				                                    test_data, indices_to_poison, train_data, 
				                                    test_description,
				                                    force_refresh=True)
   
            poisoned_X_train_subset = poison_with_influence_proj_gradient_step(
                full_model, 
                eff_indices_to_poison,
                grad_influence_wrt_input_val_subset,
                step_size,
                project_fn)


