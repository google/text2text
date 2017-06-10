# Copyright 2017 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""library codes copied from elsewhere for customization.
"""
import random

import tensorflow as tf
from tensorflow.python.framework import tensor_shape


def _reverse_seq(input_seq, lengths):
  """Reverse a list of Tensors up to specified lengths.

  Args:
    input_seq: Sequence of seq_len tensors of dimension (batch_size, depth)
    lengths:   A tensor of dimension batch_size, containing lengths for each
               sequence in the batch. If "None" is specified, simply reverses
               the list.

  Returns:
    time-reversed sequence
  """
  if lengths is None:
    return list(reversed(input_seq))

  input_shape = tensor_shape.matrix(None, None)
  for input_ in input_seq:
    input_shape.merge_with(input_.get_shape())
    input_.set_shape(input_shape)

  # Join into (time, batch_size, depth)
  s_joined = tf.stack(input_seq)

  if lengths is not None:
    lengths = tf.to_int64(lengths)

  # Reverse along dimension 0
  s_reversed = tf.reverse_sequence(s_joined, lengths, 0, 1)
  # Split again into list
  result = tf.unstack(s_reversed)
  for r in result:
    r.set_shape(input_shape)
  return result


def bidirectional_rnn(cell_fw, cell_bw, inputs,
                      initial_state_fw=None, initial_state_bw=None,
                      dtype=None, sequence_length=None, scope=None):
  """Creates a bidirectional recurrent neural network.

  Similar to the unidirectional case above (rnn) but takes input and builds
  independent forward and backward RNNs with the final forward and backward
  outputs depth-concatenated, such that the output will have the format
  [time][batch][cell_fw.output_size + cell_bw.output_size]. The input_size of
  forward and backward cell must match. The initial state for both directions
  is zero by default (but can be set optionally) and no intermediate states are
  ever returned -- the network is fully unrolled for the given (passed in)
  length(s) of the sequence(s) or completely unrolled if length(s) is not given.

  Args:
    cell_fw: An instance of RNNCell, to be used for forward direction.
    cell_bw: An instance of RNNCell, to be used for backward direction.
    inputs: A length T list of inputs, each a tensor of shape
      [batch_size, input_size].
    initial_state_fw: (optional) An initial state for the forward RNN.
      This must be a tensor of appropriate type and shape
      `[batch_size x cell_fw.state_size]`.
      If `cell_fw.state_size` is a tuple, this should be a tuple of
      tensors having shapes `[batch_size, s] for s in cell_fw.state_size`.
    initial_state_bw: (optional) Same as for `initial_state_fw`, but using
      the corresponding properties of `cell_bw`.
    dtype: (optional) The data type for the initial state.  Required if
      either of the initial states are not provided.
    sequence_length: (optional) An int32/int64 vector, size `[batch_size]`,
      containing the actual lengths for each of the sequences.
    scope: VariableScope for the created subgraph; defaults to "BiRNN"

  Returns:
    A tuple (outputs, output_state_fw, output_state_bw) where:
      outputs is a length `T` list of outputs (one for each input), which
        are depth-concatenated forward and backward outputs.
      output_state_fw is the final state of the forward rnn.
      output_state_bw is the final state of the backward rnn.

  Raises:
    TypeError: If `cell_fw` or `cell_bw` is not an instance of `RNNCell`.
    ValueError: If inputs is None or an empty list.
  """

  if not isinstance(cell_fw, tf.contrib.rnn.RNNCell):
    raise TypeError('cell_fw must be an instance of RNNCell')
  if not isinstance(cell_bw, tf.contrib.rnn.RNNCell):
    raise TypeError('cell_bw must be an instance of RNNCell')
  if not isinstance(inputs, list):
    raise TypeError('inputs must be a list')
  if not inputs:
    raise ValueError('inputs must not be empty')

  name = scope or 'BiRNN'
  # Forward direction
  with tf.variable_scope(name + '_FW') as fw_scope:
    output_fw, output_state_fw = tf.contrib.rnn.static_rnn(
        cell_fw, inputs, initial_state_fw, dtype,
        sequence_length, scope=fw_scope)

  # Backward direction
  with tf.variable_scope(name + '_BW') as bw_scope:
    tmp, output_state_bw = tf.contrib.rnn.static_rnn(
        cell_bw,
        _reverse_seq(inputs, sequence_length),
        initial_state_bw,
        dtype,
        sequence_length,
        scope=bw_scope)
  output_bw = _reverse_seq(tmp, sequence_length)
  # Concat each of the forward/backward outputs
  outputs = [tf.concat([fw, bw], 1) for fw, bw in zip(output_fw, output_bw)]
  return (outputs, output_state_fw, output_state_bw)



# Adapted to support sampled_softmax loss function, which accets activations
# instead of logits.
def sequence_loss_by_example(inputs, targets, weights, loss_function,
                             average_across_timesteps=True, name=None):
  """Sampled softmax loss for a sequence of inputs (per example).

  Args:
    inputs: List of 2D Tensors of shape [batch_size x hid_dim].
    targets: List of 1D batch-sized int32 Tensors of the same length as logits.
    weights: List of 1D batch-sized float-Tensors of the same length as logits.
    loss_function: Sampled softmax function (labels, inputs) -> loss
    average_across_timesteps: If set, divide the returned cost by the total
      label weight.
    name: Optional name for this operation, default: 'sequence_loss_by_example'.

  Returns:
    1D batch-sized float Tensor: The log-perplexity for each sequence.

  Raises:
    ValueError: If len(inputs) is different from len(targets) or len(weights).
  """
  if len(targets) != len(inputs) or len(weights) != len(inputs):
    raise ValueError('Lengths of logits, weights, and targets must be the same '
                     '%d, %d, %d.' % (len(inputs), len(weights), len(targets)))
  with tf.name_scope(
      name,
      'sequence_loss_by_example',
      inputs + targets + weights,):
    log_perp_list = []
    for inp, target, weight in zip(inputs, targets, weights):
      crossent = loss_function(target, inp)
      log_perp_list.append(crossent * weight)
    log_perps = tf.add_n(log_perp_list)
    if average_across_timesteps:
      total_size = tf.add_n(weights)
      total_size += 1e-12  # Just to avoid division by 0 for all-0 weights.
      log_perps /= total_size
  return log_perps


def sampled_sequence_loss(inputs, targets, weights, loss_function,
                          average_across_timesteps=True,
                          average_across_batch=True, name=None):
  """Weighted cross-entropy loss for a sequence of logits, batch-collapsed.

  Args:
    inputs: List of 2D Tensors of shape [batch_size x hid_dim].
    targets: List of 1D batch-sized int32 Tensors of the same length as inputs.
    weights: List of 1D batch-sized float-Tensors of the same length as inputs.
    loss_function: Sampled softmax function (labels, inputs) -> loss
    average_across_timesteps: If set, divide the returned cost by the total
      label weight.
    average_across_batch: If set, divide the returned cost by the batch size.
    name: Optional name for this operation, defaults to 'sequence_loss'.

  Returns:
    A scalar float Tensor: The average log-perplexity per symbol (weighted).

  Raises:
    ValueError: If len(inputs) is different from len(targets) or len(weights).
  """
  with tf.name_scope(name, 'sampled_sequence_loss', inputs + targets + weights):
    cost = tf.reduce_sum(sequence_loss_by_example(
        inputs, targets, weights, loss_function,
        average_across_timesteps=average_across_timesteps))
    if average_across_batch:
      batch_size = tf.shape(targets[0])[0]
      return cost / tf.cast(batch_size, tf.float32)
    return cost


def bow_loss(logits,
             targets,
             weights,
             average_across_timesteps=False,
             average_across_batch=True):
  """Loss for a bow of logits.

  As opposed to sequence loss this is supposed to ignore the order.
  Does not seem to work yet.

  Args:
    logits: List of 2D Tensors of shape [batch_size x num_decoder_symbols].
    targets: List of 1D batch-sized int32 Tensors of the same length as logits.
    weights: List of 1D batch-sized float-Tensors of the same length as logits.
    average_across_timesteps: If set, divide the returned cost by the total
        label weight.
    average_across_batch: If set, divide the returned cost by the batch size.
        to be used instead of the standard softmax (the default if this is
        None).

  Returns:
    A scalar float Tensor: The average loss per symbol (weighted).

  Raises:
    ValueError: If len(logits) is different from len(targets) or len(weights).
  """

  cost = tf.reduce_sum(
      bow_loss_by_example(
          logits,
          targets,
          weights,
          average_across_timesteps=average_across_timesteps),
      axis=0)
  if average_across_batch:
    batch_size = targets[0].shape[0]
    return cost / tf.cast(batch_size, cost.dtype)
  return cost


def bow_loss_by_example(logits,
                        targets,
                        weights,
                        average_across_timesteps=False):
  """Loss for a bow of logits (per example).

  As opposed to sequence loss this is supposed to ignore the order.
  Does not seem to work yet.

  Args:
    logits: List of 2D Tensors of shape [batch_size x num_decoder_symbols].
    targets: List of 1D batch-sized int32 Tensors of the same length as
      logits.
    weights: List of 1D batch-sized float-Tensors of the same length as
      logits.
    average_across_timesteps: If set, divide the returned cost by the total
      label weight.

  Returns:
    1D batch-sized float Tensor: The loss for each bow.

  Raises:
    ValueError: If len(logits) is different from len(targets) or len(weights).
  """
  if len(targets) != len(logits) or len(weights) != len(logits):
    raise ValueError('Lengths of logits, weights, and targets must be the same '
                     '%d, %d, %d.' % (len(logits), len(weights), len(targets)))

  batch_size = logits[0].shape[0]
  vocab_size = logits[0].shape[1]

  logitssum = tf.zeros((batch_size, vocab_size), tf.float32)
  targetset = tf.zeros((batch_size, vocab_size), tf.float32)
  for target, weight in zip(targets, weights):
    targetset += (tf.one_hot(target, vocab_size) * weight[:, None])
  weight = tf.ones((batch_size), tf.float32)
  for logit in logits:
    softmax = tf.nn.softmax(logit)
    logitssum += (logitssum * weight[:, None])
    weight = tf.maximum(0.0, weight - softmax[:, 3])

  # logitssum = tf.minimum(logitssum, 1.0)
  # targetset = tf.minimum(targetset, 1.0)
  # loss = tf.nn.sigmoid_cross_entropy_with_logits(
  #     labels=targetset, logits=logitssum)

  loss = tf.reduce_sum(tf.squared_difference(logitssum, targetset), axis=1)

  # crossent = tf.maximum(logitssum, 0.0) - (
  #     logitssum * targetset) + tf.log(1.0 + tf.exp(-1.0 * tf.abs(logitssum)))
  # log_perps = tf.reduce_logsumexp(crossent, axis=1)

  if average_across_timesteps:
    total_size = tf.add_n(weights)
    total_size += 1e-12  # Just to avoid division by 0 for all-0 weights.
    loss /= total_size

  return loss


def linear(args, output_size, bias, bias_start=0.0, scope=None):
  """Linear map: sum_i(args[i] * W[i]), where W[i] is a variable.

  Args:
    args: a 2D Tensor or a list of 2D, batch x n, Tensors.
    output_size: int, second dimension of W[i].
    bias: boolean, whether to add a bias term or not.
    bias_start: starting value to initialize the bias; 0 by default.
    scope: VariableScope for the created subgraph; defaults to "Linear".

  Returns:
    A 2D Tensor with shape [batch x output_size] equal to
    sum_i(args[i] * W[i]), where W[i]s are newly created matrices.

  Raises:
    ValueError: if some of the arguments has unspecified or wrong shape.
  """
  if args is None or (isinstance(args, (list, tuple)) and not args):
    raise ValueError('`args` must be specified')
  if not isinstance(args, (list, tuple)):
    args = [args]

  # Calculate the total size of arguments on dimension 1.
  total_arg_size = 0
  shapes = [a.get_shape().as_list() for a in args]
  for shape in shapes:
    if len(shape) != 2:
      raise ValueError('Linear is expecting 2D arguments: %s' % str(shapes))
    if not shape[1]:
      raise ValueError('Linear expects shape[1] of arguments: %s' % str(shapes))
    else:
      total_arg_size += shape[1]

  # Now the computation.
  with tf.variable_scope(scope or 'Linear'):
    matrix = tf.get_variable('Matrix', [total_arg_size, output_size])
    if len(args) == 1:
      res = tf.matmul(args[0], matrix)
    else:
      res = tf.matmul(tf.concat(args, 1), matrix)
    if not bias:
      return res
    bias_term = tf.get_variable(
        'Bias', [output_size],
        initializer=tf.constant_initializer(bias_start))
  return res + bias_term


def sequence_loss_by_example_cg(inputs_gen,
                                inputs_cop,
                                targets_gen,
                                targets_cop,
                                is_copy_weights,
                                weights,
                                loss_function_g,
                                loss_function_c,
                                switch_gate_ignore_prob,
                                is_copy_when_switch_gate_ignored,
                                unk_token_id,
                                unk_token_penalty,
                                average_across_timesteps=True,
                                name=None):
  """Sampled softmax loss for a sequence of inputs (per example).

  Args:
    inputs_gen: List of 2D Tensors of shape [batch_size x hid_dim] from g-mode
    inputs_cop: List of 2D Tensors of shape [batch_size x hid_dim] from c-mode
    targets_gen: List of 1D batch-sized int32 Tensors of the same length as
    inputs in g-mode
    targets_cop: List of 1D batch-sized int32 Tensors of the same length as
    inputs in c-mode
    is_copy_weights: List of 1D batch-sized float32 Tensors of the same length
    as inputs as output of switch gate
    weights: List of 1D batch-sized float-Tensors of the same length as inputs.
    loss_function_g: loss_function: (sampled) softmax function for g-mode
    (labels, inputs) -> loss
    loss_function_c: loss_function: (sampled) softmax function for c-mode
    (labels, inputs) -> loss
    switch_gate_ignore_prob: probability of ignoring switch gate for
    calulating loss
    is_copy_when_switch_gate_ignored: share of copier loss when switch gate
    is ignored
    unk_token_id: id of the <UNK> token
    unk_token_penalty: penalty for predicting <UNK> token
    average_across_timesteps: If set, divide the returned cost by the total
      label weight.
    name: Optional name for this operation, defaults to 'sequence_loss'.

  Returns:
    1D batch-sized float Tensor: The log-perplexity for each sequence.

  Raises:
    ValueError: If len(inputs) is different from len(targets) or len(weights).
  """
  if (len(targets_gen) != len(inputs_gen) or
      len(targets_cop) != len(inputs_cop) or len(weights) != len(targets_cop) or
      len(weights) != len(targets_gen)):
    raise ValueError('Lengths of logits_gen, logits_cop, weights, '
                     'targets_gen, targets_cop must be the same '
                     '%d, %d, %d, %d, %d.' %
                     (len(inputs_gen), len(inputs_cop), len(weights),
                      len(targets_gen), len(targets_cop)))
  with tf.name_scope(
      name,
      'sequence_loss_by_example',
      inputs_gen + inputs_cop + targets_gen + targets_cop + weights,):

    log_perp_list = []
    for (inp_g, inp_c, target_g, target_c, is_copy_weight, weight) in zip(
            inputs_gen, inputs_cop, targets_gen, targets_cop, is_copy_weights,
            weights):
      # loss of generator
      crossent_g = loss_function_g(target_g, inp_g)
      # loss of copier
      crossent_c = loss_function_c(target_c, inp_c)

      # switch gate prob:
      is_copy_weight = tf.squeeze(is_copy_weight)

      # linearly weighted loss with dropout:
      switch_gate_use_prob = random.uniform(0, 1)
      if switch_gate_use_prob > switch_gate_ignore_prob:
        crossent = (is_copy_weight * crossent_c) + (
            (1 - is_copy_weight) * crossent_g)
      else:
        crossent = (is_copy_when_switch_gate_ignored * crossent_c +
                    (1 - is_copy_when_switch_gate_ignored) * crossent_g)

      # penalizing if the selected model by the switch gate predicts "<unk>".
      predicted_gen = tf.argmax(inp_g, 1)  # assuming we don't have sampling
      predicted_cop = tf.argmax(inp_c, 1)

      crossent = tf.where(predicted_gen == unk_token_id, crossent +
                          (1 - is_copy_weight) * unk_token_penalty, crossent)
      crossent = tf.where(predicted_cop == unk_token_id, crossent +
                          (is_copy_weight) * unk_token_penalty, crossent)

      log_perp_list.append(crossent * weight)

    log_perps = tf.add_n(log_perp_list)
    if average_across_timesteps:
      total_size = tf.add_n(weights)
      total_size += 1e-12  # Just to avoid division by 0 for all-0 weights.
      log_perps /= total_size
  return log_perps


def sequence_loss_cg(inputs_gen,
                     inputs_cop,
                     targets_gen,
                     targets_cop,
                     is_copy_weights,
                     weights,
                     loss_function_g,
                     loss_function_c,
                     switch_gate_ignore_prob,
                     is_copy_when_switch_gate_ignored,
                     unk_token_id,
                     unk_token_penalty,
                     loss_type='',
                     average_across_timesteps=True,
                     average_across_batch=True,
                     name=None):
  """Weighted cross-entropy loss for a sequence of logits, batch-collapsed.

  Args:
    inputs_gen: List of 2D Tensors of shape [batch_size x hid_dim] from g-mode
    inputs_cop: List of 2D Tensors of shape [batch_size x hid_dim] from c-mode
    targets_gen: List of 1D batch-sized int32 Tensors of the same length as
    inputs in g-mode
    targets_cop: List of 1D batch-sized int32 Tensors of the same length as
    inputs in c-mode
    is_copy_weights: List of 1D batch-sized float32 Tensors of the same length
    as inputs as output of switch gate
    weights: List of 1D batch-sized float-Tensors of the same length as inputs.
    loss_function_g: loss_function: (sampled) softmax function for g-mode
    (labels, inputs) -> loss
    loss_function_c: loss_function: (sampled) softmax function for c-mode
    (labels, inputs) -> loss
    switch_gate_ignore_prob: probability of ignoring switch gate for
    calulating loss
    is_copy_when_switch_gate_ignored: share of copier loss when switch gate
    is ignored
    unk_token_id: id of the <UNK> token
    unk_token_penalty: penalty for predicting <UNK> token
    loss_type: "sampled" or ""
    average_across_timesteps: If set, divide the returned cost by the total
      label weight.
    average_across_batch: If set, divide the returned cost by the batch size.
    name: Optional name for this operation, defaults to 'sequence_loss'.

  Returns:
    A scalar float Tensor: The average log-perplexity per symbol (weighted).

  Raises:
    ValueError: If len(inputs) is different from len(targets) or len(weights).
  """
  with tf.name_scope(
      name, loss_type + '_sequence_loss',
      inputs_gen + inputs_cop + targets_gen + targets_cop + weights):
    cost = tf.reduce_sum(
        sequence_loss_by_example_cg(
            inputs_gen,
            inputs_cop,
            targets_gen,
            targets_cop,
            is_copy_weights,
            weights,
            loss_function_g,
            loss_function_c,
            switch_gate_ignore_prob,
            is_copy_when_switch_gate_ignored,
            unk_token_id,
            unk_token_penalty,
            average_across_timesteps=average_across_timesteps))
    if average_across_batch:
      batch_size = tf.shape(targets_gen[0])[0]
      return cost / tf.cast(batch_size, cost.dtype)  # tf.float32
    return cost
