
class Model(object):
  """docstring for Model"""
  # def __init__(self, arg):
  #   super(Model, self).__init__()
  #   self.model_name = # Name of model

  def grad_loss_no_reg(x, y):
    # self.sess.run(grad_loss_no_reg_op, feed_dict={input_placeholder=x, label_placeholder=y})
    # return gradient of loss

  def grad_total_loss(data, labels):
    # compute the gradient of the total loss (w/ reg val) on the 
    # input data w.r.t model parameters

  def hessian_vector_product(train_dataset, vector):
    # use total loss graph and parameters Tf.variables and multiply them by a vector

  def params():
    # return all the parameters in the model

  def grad_influence_wrt_input(inverse_hvp, xTr, yTr):
    # inverse_hvp is the product of:
    # 1. The Hessian of the total training loss with respect to the parameters (P x P)
    # 2. The gradient of the test loss with respect to the parameters (P x 1)
    # xTr and yTr are the training points for which we get the perturbation influence.
    # We do this by:
    # 1. computing the product of inverse_hvp (P x 1) with the gradient of the loss of (xTr, yTr) w.r.t the parameters (P x 1)
    # 2. Taking the gradient of the product w.r.t. xTr
  
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

# product of hessian matrix of [model's loss on train_dataset with respect to parameters] with x 
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

# Gradients is a list of size P (for P paramaters in the model)
# Needs train_data to get loss to compute Hessian of model's loss w.r.t parameters
# Uses Conjugate Gradient approximation of HVP, see section 3 subsection Conjugate Gradient 
# to see what they are trying to do here
# minibatch_hessian_vector_val(x, model, train_dataset, batch_size = 100, damping=0)
def get_inverse_hvp_cg(model, train_data, gradients, verbose=True):
  params_val = model.params
  num_params = len(np.concatenate(params_val))        
  print('Total number of parameters: %s' % num_params)
  def vec_to_list(v):
    return_list = []
    cur_pos = 0
    for p in params_val:
      return_list.append(v[cur_pos : cur_pos+len(p)])
      cur_pos += len(p)
    assert cur_pos == len(v)
    return return_list
  def get_fmin_hvp(self, x, p):
    hessian_vector_val = minibatch_hessian_vector_val(vec_to_list(p), model, train_data)
    return np.concatenate(hessian_vector_val)
  # return the loss function: .5 t^T * H * t - v^T * t
  # v is a list of parameter gradients of size P (for P parameters in the model)
  def get_fmin_loss(t):
    hessian_vector_val = minibatch_hessian_vector_val(vec_to_list(t), model, train_data)
    # np.concatenate(gradients) turns a list of P parameters into a tensor of axis=0 size P
    return 0.5 * np.dot(np.concatenate(hessian_vector_val), t) - np.dot(np.concatenate(gradients), t)
  def get_fmin_grad(t):
    hessian_vector_val = minibatch_hessian_vector_val(vec_to_list(t), model, train_data)
    return np.concatenate(hessian_vector_val) - np.concatenate(gradients)
  fmin_loss_fn = get_fmin_loss
  fmin_grad_fn = get_fmin_grad
  # cg_callback = self.get_cg_callback(gradients, verbose)
  fmin_results = fmin_ncg(
      f=fmin_loss_fn,
      x0=np.concatenate(gradients),
      fprime=fmin_grad_fn,
      fhess_p=self.get_fmin_hvp,
      # callback=cg_callback,
      avextol=1e-8,
      maxiter=100) 
  return vec_to_list(fmin_results)

def get_influence_on_test_loss(model, 
                               test_data, train_idx, train_data, 
                               test_description,
                               force_refresh=True):
  # If train_idx is None then use X and Y (phantom points)
  # Need to make sure test_data stays consistent between models
  # because mini-batching permutes dataset order

  # returns a list of gradients of size P (to represent P parameters in the model)
  test_grad_loss_no_reg_val = get_test_grad_loss_no_reg_val(model, test_data)
  print('Norm of test gradient: %s' % np.linalg.norm(np.concatenate(test_grad_loss_no_reg_val)))

  # The approximation of the product (Hession matrix & gradients of test loss) is 
  # by the Conjugate Gradiant approximation and the result is stored so that 
  # no future computation will be needed
  start_time = time.time()
  approximation_filename = os.path.join(self.train_dir, '%s-test-%s.npz' % (model.model_name, test_description))
  if os.path.exists(approximation_filename) and force_refresh == False:
      inverse_hvp = list(np.load(approximation_filename)['inverse_hvp'])
      print('Loaded inverse HVP from %s' % approximation_filename)
  else:
      inverse_hvp = get_inverse_hvp_cg(model, train_data, test_grad_loss_no_reg_val)
      np.savez(approximation_filename, inverse_hvp=inverse_hvp)
      print('Saved inverse HVP to %s' % approximation_filename)
  duration = time.time() - start_time
  print('Inverse HVP took %s sec' % duration)

  start_time = time.time()
  num_to_remove = len(train_idx)
  predicted_loss_diffs = np.zeros([num_to_remove])
  for counter, idx_to_remove in enumerate(train_idx):            
    train_grad_loss_val = model.grad_total_loss(train_data.x[idx_to_remove, :].reshape(1, -1),
                                                train_data.labels[idx_to_remove].reshape(-1))
    predicted_loss_diffs[counter] = np.dot(np.concatenate(inverse_hvp), 
                                           np.concatenate(train_grad_loss_val)) / len(train_data)
  duration = time.time() - start_time
  print('Multiplying by %s train examples took %s sec' % (num_to_remove, duration))

  return predicted_loss_diffs


def get_grad_of_influence_wrt_input(model, 
                                    test_data, train_idx, train_data, 
                                    test_description,
                                    force_refresh=True):
  """
  If the loss goes up when you remove a point, then it was a helpful point.
  So positive influence = helpful.
  If we move in the direction of the gradient, we make the influence even more positive, 
  so even more helpful.
  Thus if we want to make the test point more wrong, we have to move in the opposite direction.
  """

  # Calculate v_placeholder (gradient of loss at test point)
  test_grad_loss_no_reg_val = get_test_grad_loss_no_reg_val(model, test_data)

  print('Norm of test gradient: %s' % np.linalg.norm(np.concatenate(test_grad_loss_no_reg_val)))
  
  start_time = time.time()

  approx_filename = os.path.join(self.train_dir, '%s-test-%s.npz' % (model.model_name, test_description))
  if os.path.exists(approx_filename) and force_refresh == False:
      inverse_hvp = list(np.load(approx_filename)['inverse_hvp'])
      print('Loaded inverse HVP from %s' % approx_filename)
  else:            
      inverse_hvp = get_inverse_hvp_cg(model, train_data, test_grad_loss_no_reg_val)
      np.savez(approx_filename, inverse_hvp=inverse_hvp)
      print('Saved inverse HVP to %s' % approx_filename)            
  duration = time.time() - start_time
  print('Inverse HVP took %s sec' % duration)

  grad_influence_wrt_input_val = None
  for counter, idx_to_remove in enumerate(train_idx):
      # Take the derivative of the influence w.r.t input       
      current_grad_influence_wrt_input_val = model.grad_influence_wrt_input(inverse_hvp, 
                                                      train_data.x[idx_to_remove, :].reshape(1, -1),
                                                      train_data.labels[idx_to_remove].reshape(-1))
      if grad_influence_wrt_input_val is None:
          grad_influence_wrt_input_val = np.zeros([len(train_idx), len(current_grad_influence_wrt_input_val)])
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
