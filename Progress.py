
class Model(object):
  """docstring for Model"""
  def __init__(self, arg):
    super(Model, self).__init__()
    self.name = # Name of model

  def grad_loss_no_reg(x, y):
    # self.sess.run(grad_loss_no_reg_op, feed_dict={input_placeholder=x, label_placeholder=y})
    # return gradient of loss

  def hessian_vector_product(train_dataset, vector)
    # use total loss graph and parameters Tf.variables and multiply them by a vector
  
# The average gradient of multiple test points' loss (to maximize mean loss for these test points)
# This is done batch-wise
# test_data.x -> N x X
# test_data.label -> N x Y
def get_test_grad_loss_no_reg_val(model, test_data, batch_size=100):
  num_iter = int(np.ceil(len(test_data) / batch_size))
  final_result = None
  for i in range(num_iter):
      start = i * batch_size
      end = int(min((i+1) * batch_size, len(test_data)))
      grad_loss_value = model.grad_loss_no_reg(test_data.x[start:end], test_data.label[start:end])
      if final_result is None:
          final_result = [next_gradient * (end-start) for next_gradient in grad_loss_value] # to undo the averaging 1/n of the loss's gradient
      else:
          final_result = [accumulated_gradient + next_gradient * (end-start) for (accumulated_gradient, next_gradient) in zip(final_result, grad_loss_value)]
  final_result = [gradient/len(test_data) for gradient in final_result] # Re-apply the 1/n in the appropriate size
  return final_result

# product of hessian matrix with x 
def minibatch_hessian_vector_val(x, model, train_dataset, batch_size = 100, damping=0):
  num_iter = int(np.ceil(len(train_dataset) / batch_size))
  train_dataset.reset_datasets()
  final_result = None
  for i in xrange(num_iter):
    start = i * batch_size
    end = int(min((i+1) * batch_size, len(train_dataset)))
    hvp_value = model.hessian_vector_product(train_dataset[start:end], x)
    if final_result is None:
      final_result = [next_hvp / float(num_iter) for next_hvp in hvp_value]
    else:
      final_result = [accumulated_hvp + (next_hvp / float(num_iter)) for (accumulated_hvp, next_hvp) in zip(final_result, hvp_value)]
  final_result = [accumulated_hvp + damping * v_vector for (accumulated_hvp, v_vector) in zip(final_result, x)]
  return final_result

def get_vec_to_list_fn(self):
  params_val = self.sess.run(self.params)
  self.num_params = len(np.concatenate(params_val))        
  print('Total number of parameters: %s' % self.num_params)
  def vec_to_list(v):
    return_list = []
    cur_pos = 0
    for p in params_val:
        return_list.append(v[cur_pos : cur_pos+len(p)])
        cur_pos += len(p)

    assert cur_pos == len(v)
    return return_list
  return vec_to_list


def get_fmin_grad_fn(self, v):
    def get_fmin_grad(x):
      hessian_vector_val = self.minibatch_hessian_vector_val(self.vec_to_list(x))
      return np.concatenate(hessian_vector_val) - np.concatenate(v)
    return get_fmin_grad

def get_cg_callback(self, v, verbose):
  fmin_loss_fn = self.get_fmin_loss_fn(v)
  def get_fmin_loss(x):
    hessian_vector_val = self.minibatch_hessian_vector_val(self.vec_to_list(x))
    return 0.5 * np.dot(np.concatenate(hessian_vector_val), x), -np.dot(np.concatenate(v), x)
  def cg_callback(x):
    # x is current params
    v = self.vec_to_list(x)
    idx_to_remove = 5
    single_train_feed_dict = self.fill_feed_dict_with_one_ex(self.data_sets.train, idx_to_remove)      
    train_grad_loss_val = self.sess.run(self.grad_total_loss_op, feed_dict=single_train_feed_dict)
    predicted_loss_diff = np.dot(np.concatenate(v), np.concatenate(train_grad_loss_val)) / self.num_train_examples
    if verbose:
        print('Function value: %s' % fmin_loss_fn(x))
        quad, lin = get_fmin_loss(x)
        print('Split function value: %s, %s' % (quad, lin))
        print('Predicted loss diff on train_idx %s: %s' % (idx_to_remove, predicted_loss_diff))
  return cg_callback

# return the loss function: .5 x^T Hx - v^Tx
def get_fmin_loss_fn(self, v):
  def get_fmin_loss(x):
    hessian_vector_val = self.minibatch_hessian_vector_val(self.vec_to_list(x))
    return 0.5 * np.dot(np.concatenate(hessian_vector_val), x) - np.dot(np.concatenate(v), x)
  return get_fmin_loss

def get_fmin_hvp(self, x, p):
  hessian_vector_val = self.minibatch_hessian_vector_val(self.vec_to_list(p))
  return np.concatenate(hessian_vector_val)

def get_inverse_hvp_cg(gradients, verbose):
  fmin_loss_fn = self.get_fmin_loss_fn(gradients)
  fmin_grad_fn = self.get_fmin_grad_fn(gradients)
  cg_callback = self.get_cg_callback(gradients, verbose)
  fmin_results = fmin_ncg(
      f=fmin_loss_fn,
      x0=np.concatenate(gradients),
      fprime=fmin_grad_fn,
      fhess_p=self.get_fmin_hvp,
      callback=cg_callback,
      avextol=1e-8,
      maxiter=100) 
  return self.vec_to_list(fmin_results)

def get_influence_on_test_loss(model, 
                               test_data, train_idx, 
                               test_description,
                               force_refresh=True):
  # If train_idx is None then use X and Y (phantom points)
  # Need to make sure test_data stays consistent between models
  # because mini-batching permutes dataset order

  test_grad_loss_no_reg_val = self.get_test_grad_loss_no_reg_val(model, test_data)
  print('Norm of test gradient: %s' % np.linalg.norm(np.concatenate(test_grad_loss_no_reg_val)))

  start_time = time.time()
  approximation_filename = os.path.join(self.train_dir, '%s-test-%s.npz' % (model.name, test_description))
  if os.path.exists(approximation_filename) and force_refresh == False:
      inverse_hvp = list(np.load(approximation_filename)['inverse_hvp'])
      print('Loaded inverse HVP from %s' % approximation_filename)
  else:
      inverse_hvp = get_inverse_hvp_cg(test_grad_loss_no_reg_val, True)
      np.savez(approximation_filename, inverse_hvp=inverse_hvp)
      print('Saved inverse HVP to %s' % approximation_filename)
  duration = time.time() - start_time
  print('Inverse HVP took %s sec' % duration)

  start_time = time.time()
  if train_idx is None:
      num_to_remove = len(Y)
      predicted_loss_diffs = np.zeros([num_to_remove])            
      for counter in np.arange(num_to_remove):
          single_train_feed_dict = self.fill_feed_dict_manual(X[counter, :], [Y[counter]])      
          train_grad_loss_val = self.sess.run(self.grad_total_loss_op, feed_dict=single_train_feed_dict)
          predicted_loss_diffs[counter] = np.dot(np.concatenate(inverse_hvp), np.concatenate(train_grad_loss_val)) / self.num_train_examples            

  else:            
      num_to_remove = len(train_idx)
      predicted_loss_diffs = np.zeros([num_to_remove])
      for counter, idx_to_remove in enumerate(train_idx):            
          single_train_feed_dict = self.fill_feed_dict_with_one_ex(self.data_sets.train, idx_to_remove)      
          train_grad_loss_val = self.sess.run(self.grad_total_loss_op, feed_dict=single_train_feed_dict)
          predicted_loss_diffs[counter] = np.dot(np.concatenate(inverse_hvp), np.concatenate(train_grad_loss_val)) / self.num_train_examples
  duration = time.time() - start_time
  print('Multiplying by %s train examples took %s sec' % (num_to_remove, duration))

  return predicted_loss_diffs


def get_grad_of_influence_wrt_input(self, train_indices, test_indices, 
  approx_type='cg', approx_params=None, force_refresh=True, verbose=True, test_description=None,
  loss_type='normal_loss'):
  """
  If the loss goes up when you remove a point, then it was a helpful point.
  So positive influence = helpful.
  If we move in the direction of the gradient, we make the influence even more positive, 
  so even more helpful.
  Thus if we want to make the test point more wrong, we have to move in the opposite direction.
  """

  # Calculate v_placeholder (gradient of loss at test point)
  test_grad_loss_no_reg_val = self.get_test_grad_loss_no_reg_val(test_indices, loss_type=loss_type)            

  if verbose: print('Norm of test gradient: %s' % np.linalg.norm(np.concatenate(test_grad_loss_no_reg_val)))
  
  start_time = time.time()

  if test_description is None:
      test_description = test_indices

  approx_filename = os.path.join(self.train_dir, '%s-%s-%s-test-%s.npz' % (self.model_name, approx_type, loss_type, test_description))
  
  if os.path.exists(approx_filename) and force_refresh == False:
      inverse_hvp = list(np.load(approx_filename)['inverse_hvp'])
      if verbose: print('Loaded inverse HVP from %s' % approx_filename)
  else:            
      inverse_hvp = self.get_inverse_hvp(
          test_grad_loss_no_reg_val,
          approx_type,
          approx_params,
          verbose=verbose)
      np.savez(approx_filename, inverse_hvp=inverse_hvp)
      if verbose: print('Saved inverse HVP to %s' % approx_filename)            
  
  duration = time.time() - start_time
  if verbose: print('Inverse HVP took %s sec' % duration)

  grad_influence_wrt_input_val = None

  for counter, train_idx in enumerate(train_indices):
      # Put in the train example in the feed dict
      grad_influence_feed_dict = self.fill_feed_dict_with_one_ex(
          self.data_sets.train,  
          train_idx)

      self.update_feed_dict_with_v_placeholder(grad_influence_feed_dict, inverse_hvp)

      # Run the grad op with the feed dict
      current_grad_influence_wrt_input_val = self.sess.run(self.grad_influence_wrt_input_op, feed_dict=grad_influence_feed_dict)[0][0, :]            
      
      if grad_influence_wrt_input_val is None:
          grad_influence_wrt_input_val = np.zeros([len(train_indices), len(current_grad_influence_wrt_input_val)])

      grad_influence_wrt_input_val[counter, :] = current_grad_influence_wrt_input_val

  return grad_influence_wrt_input_val


def get_project_fn(X_orig, box_radius_in_pixels=0.5):
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

def hessian_vector_product(ys, xs, v):
  """Multiply the Hessian of `ys` wrt `xs` by `v`.
  This is an efficient construction that uses a backprop-like approach
  to compute the product between the Hessian and another vector. The
  Hessian is usually too large to be explicitly computed or even
  represented, but this method allows us to at least multiply by it
  for the same big-O cost as backprop.
  Implicit Hessian-vector products are the main practical, scalable way
  of using second derivatives with neural networks. They allow us to
  do things like construct Krylov subspaces and approximate conjugate
  gradient descent.
  Example: if `y` = 1/2 `x`^T A `x`, then `hessian_vector_product(y,
  x, v)` will return an expression that evaluates to the same values
  as (A + A.T) `v`.
  Args:
    ys: A scalar value, or a tensor or list of tensors to be summed to
        yield a scalar.
    xs: A list of tensors that we should construct the Hessian over.
    v: A list of tensors, with the same shapes as xs, that we want to
       multiply by the Hessian.
  Returns:
    A list of tensors (or if the list would be length 1, a single tensor)
    containing the product between the Hessian and `v`.
  Raises:
    ValueError: `xs` and `v` have different length.
  """ 

  # Validate the input
  length = len(xs)
  if len(v) != length:
    raise ValueError("xs and v must have the same length.")

  # First backprop
  grads = gradients(ys, xs)

  # grads = xs

  assert len(grads) == length

  elemwise_products = [
      math_ops.multiply(grad_elem, array_ops.stop_gradient(v_elem))
      for grad_elem, v_elem in zip(grads, v) if grad_elem is not None
  ]

  # Second backprop  
  grads_with_none = gradients(elemwise_products, xs)
  return_grads = [
      grad_elem if grad_elem is not None \
      else tf.zeros_like(x) \
      for x, grad_elem in zip(xs, grads_with_none)]
  
  return return_grads