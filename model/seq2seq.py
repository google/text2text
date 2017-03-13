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

"""Sequence-to-Sequence RNNs and Beyond.

Paper: https://arxiv.org/abs/1602.06023
"""

import numpy as np
import tensorflow as tf
import library

class Model(object):
  """Wrapper for Tensorflow model graph for text sum vectors."""

  def __init__(self, config, mode, num_gpus=0):
    self._config = config
    self._mode = mode
    self._input_vocab = config.input_vocab
    self._output_vocab = config.output_vocab
    self._num_gpus = num_gpus
    self._cur_gpu = 0

  def run_train_step(self, sess, encoder_inputs, decoder_inputs, targets,
                     encoder_len, decoder_len, loss_weights):
    to_return = [self._train_op, self._summaries, self._loss, self.global_step]
    return sess.run(
        to_return,
        feed_dict={
            self._encoder_inputs: encoder_inputs,
            self._decoder_inputs: decoder_inputs,
            self._targets: targets,
            self._encoder_len: encoder_len,
            self._decoder_len: decoder_len,
            self._loss_weights: loss_weights
        })

  def run_eval_step(self, sess, encoder_inputs, decoder_inputs, targets,
                    encoder_len, decoder_len, loss_weights):
    to_return = [self._summaries, self._loss, self.global_step]
    return sess.run(
        to_return,
        feed_dict={
            self._encoder_inputs: encoder_inputs,
            self._decoder_inputs: decoder_inputs,
            self._targets: targets,
            self._encoder_len: encoder_len,
            self._decoder_len: decoder_len,
            self._loss_weights: loss_weights
        })

  def _next_device(self):
    """Round robin the gpu device. (Reserve last gpu for expensive op)."""
    if self._num_gpus == 0:
      return ''
    dev = '/gpu:%d' % self._cur_gpu
    self._cur_gpu = (self._cur_gpu + 1) % (self._num_gpus - 1)
    return dev

  def _get_gpu(self, gpu_id):
    if self._num_gpus <= 0 or gpu_id >= self._num_gpus:
      return ''
    return '/gpu:%d' % gpu_id

  def _add_placeholders(self):
    """Inputs to be fed to the graph."""
    config = self._config

    # Only need to restore the 1st step and reuse it since
    # we keep and feed in state for each step's output.
    if self._mode == 'decode':
      max_output_len = 1
    else:
      max_output_len = config.max_output_len

    self._encoder_inputs = tf.placeholder(
        tf.int32, [config.batch_size, config.max_input_len],
        name='encoder_inputs')
    self._decoder_inputs = tf.placeholder(
        tf.int32, [config.batch_size, max_output_len], name='decoder_inputs')
    self._targets = tf.placeholder(
        tf.int32, [config.batch_size, max_output_len], name='targets')
    self._encoder_len = tf.placeholder(
        tf.int32, [config.batch_size], name='encoder_len')
    self._decoder_len = tf.placeholder(
        tf.int32, [config.batch_size], name='decoder_len')
    self._loss_weights = tf.placeholder(
        tf.float32, [config.batch_size, max_output_len], name='loss_weights')

  def _add_seq2seq(self):
    """Graph for the model."""
    config = self._config

    with tf.variable_scope('seqgen'):
      encoder_inputs = tf.unstack(self._encoder_inputs, axis=1)
      decoder_inputs = tf.unstack(self._decoder_inputs, axis=1)
      targets = tf.unstack(self._targets, axis=1)
      loss_weights = tf.unstack(self._loss_weights, axis=1)
      encoder_len = self._encoder_len

      with tf.variable_scope('embedding'), tf.device('/cpu:0'):
        input_vsize = self._input_vocab.NumIds()
        input_embedding = tf.get_variable(
            'embedding', [input_vsize, config.emb_dim],
            dtype=tf.float32,
            initializer=tf.truncated_normal_initializer(stddev=1e-4))
        # Embedding shared by the input and outputs.
        if self._output_vocab == self._input_vocab:
          output_embedding = input_embedding
          output_vsize = input_vsize
          tf.logging.info('Using same matrix for input and output. Size: %d',
                          output_vsize)
        # Different embedding matrix for input and output
        # Use same vocab file twice if you want this but have only one vocab
        else:
          output_vsize = self._output_vocab.NumIds()
          output_embedding = tf.get_variable(
              'output_embedding', [output_vsize, config.emb_dim],
              dtype=tf.float32,
              initializer=tf.truncated_normal_initializer(stddev=1e-4))
          tf.logging.info(
              'Using different matrices for input and output. Sizes: %d and %d',
              input_vsize, output_vsize)
        emb_encoder_inputs = [
            tf.nn.embedding_lookup(input_embedding, x) for x in encoder_inputs
        ]
        emb_decoder_inputs = [
            tf.nn.embedding_lookup(output_embedding, x) for x in decoder_inputs
        ]

      for layer_i in range(config.enc_layers):
        with tf.variable_scope('encoder%d' %
                               layer_i), tf.device(self._next_device()):
          cell_fw = tf.contrib.rnn.LSTMCell(
              config.num_hidden,
              initializer=tf.random_uniform_initializer(
                  -0.1, 0.1, seed=config.random_seed + 2),
              state_is_tuple=False)
          cell_bw = tf.contrib.rnn.LSTMCell(
              config.num_hidden,
              initializer=tf.random_uniform_initializer(
                  -0.1, 0.1, seed=config.random_seed + 3),
              state_is_tuple=False)
          (emb_encoder_inputs, fw_state, _) = library.bidirectional_rnn(
              cell_fw,
              cell_bw,
              emb_encoder_inputs,
              dtype=tf.float32,
              sequence_length=encoder_len)
      encoder_outputs = emb_encoder_inputs

      with tf.variable_scope('output_projection'):
        output_bias = tf.get_variable(
            'output_bias', [output_vsize],
            dtype=tf.float32,
            initializer=tf.truncated_normal_initializer(stddev=1e-4))

      with tf.variable_scope('decoder'), tf.device(self._next_device()):
        cell = tf.contrib.rnn.LSTMCell(
            config.num_hidden,
            initializer=tf.random_uniform_initializer(
                -0.1, 0.1, seed=config.random_seed + 5),
            state_is_tuple=False)

        encoder_outputs = [
            tf.reshape(x, [config.batch_size, 1, 2 * config.num_hidden])
            for x in encoder_outputs
        ]
        self._enc_top_states = tf.concat(encoder_outputs, 1)
        self._dec_in_state = fw_state

        # Always use initial state attention because we need it for beam search
        initial_state_attention = True
        decoder_outputs, self._dec_out_state = (
            tf.contrib.legacy_seq2seq.attention_decoder(
                emb_decoder_inputs,
                self._dec_in_state,
                self._enc_top_states,
                cell,
                output_size=config.emb_dim,
                num_heads=config.num_heads,
                loop_function=None,
                initial_state_attention=initial_state_attention))

      with tf.variable_scope('output'), tf.device(self._next_device()):
        model_outputs = []
        for i in range(len(decoder_outputs)):
          if i > 0:
            tf.get_variable_scope().reuse_variables()
          model_outputs.append(
              tf.nn.xw_plus_b(decoder_outputs[i],
                              tf.transpose(output_embedding), output_bias))

      if self._mode == 'decode':
        with tf.variable_scope('decode_output'), tf.device('/cpu:0'):
          best_outputs = [tf.argmax(x, 1) for x in model_outputs]
          self._outputs = tf.concat(
              [tf.reshape(x, [config.batch_size, 1]) for x in best_outputs], 1)
          self._topk_log_probs, self._topk_ids = tf.nn.top_k(
              tf.log(tf.nn.softmax(model_outputs[-1])), config.batch_size * 2)

      with tf.variable_scope('loss'), tf.device(self._next_device()):

        def sampled_loss_func(labels, inputs):
          with tf.device('/cpu:0'):  # Try gpu.
            labels = tf.reshape(labels, [-1, 1])
            return tf.nn.sampled_softmax_loss(
                weights=output_embedding,
                biases=output_bias,
                labels=labels,
                inputs=inputs,
                num_sampled=config.num_softmax_samples,
                num_classes=output_vsize)

        if config.num_softmax_samples != 0 and self._mode == 'train':
          self._loss = library.sampled_sequence_loss(
              decoder_outputs, targets, loss_weights, sampled_loss_func)
        else:
          self._loss = tf.contrib.legacy_seq2seq.sequence_loss(
              model_outputs, targets, loss_weights)

    tf.summary.scalar('loss/min12', tf.minimum(12.0, self._loss))
    tf.summary.scalar('loss/unclamped', self._loss)

  def _add_train_op(self):
    """Sets self._train_op, op to run for training."""
    config = self._config

    tvars = tf.trainable_variables()
    with tf.device(self._get_gpu(self._num_gpus - 1)):
      grads, global_norm = tf.clip_by_global_norm(
          tf.gradients(self._loss, tvars), config.max_grad_norm)
    tf.summary.scalar('global_norm', global_norm)

    lr_rate = tf.maximum(
        config.min_lr,  # min_lr_rate.
        tf.train.exponential_decay(config.lr, self.global_step, 30000, 0.98))

    if config.optimizer == 'adam':
      optimizer = tf.train.AdamOptimizer(lr_rate, epsilon=config.adam_epsilon)
    else:
      assert config.optimizer == 'gradient_descent', config.optimizer
      optimizer = tf.train.GradientDescentOptimizer(lr_rate)

    tf.summary.scalar('learning_rate', lr_rate)
    self._train_op = optimizer.apply_gradients(
        zip(grads, tvars), global_step=self.global_step, name='train_step')

  def encode_top_state(self, sess, enc_inputs, enc_len):
    """Return the top states from encoder for decoder.

    Args:
      sess: tensorflow session.
      enc_inputs: encoder inputs of shape [batch_size, max_input_len].
      enc_len: encoder input length of shape [batch_size]
    Returns:
      enc_top_states: The top level encoder states.
      dec_in_state: The decoder layer initial state.
    """
    results = sess.run(
        [self._enc_top_states, self._dec_in_state],
        feed_dict={
            self._encoder_inputs: enc_inputs,
            self._encoder_len: enc_len
        })
    return results[0], results[1][0]

  def decode_topk(self, sess, latest_tokens, enc_top_states, dec_init_states,
                  _):
    """Return the topK results and new decoder states."""
    feed = {
        self._enc_top_states: enc_top_states,
        self._dec_in_state: np.squeeze(np.array(dec_init_states)),
        self._decoder_inputs: np.transpose(np.array([latest_tokens])),
        self._decoder_len: np.ones([len(dec_init_states)], np.int32)
    }

    results = sess.run(
        [self._topk_ids, self._topk_log_probs, self._dec_out_state],
        feed_dict=feed)

    ids, probs, states = results[0], results[1], results[2]
    new_states = [s for s in states]
    return ids, probs, new_states

  def build_graph(self):
    self._add_placeholders()
    self._add_seq2seq()
    self.global_step = tf.Variable(0, name='global_step', trainable=False)
    if self._mode == 'train':
      self._add_train_op()
    self._summaries = tf.summary.merge_all()
