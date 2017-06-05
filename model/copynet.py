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

"""Implementation of copynet.

Inspired by: "https://arxiv.org/abs/1603.06393"

NOTE:
  We have two modes, generation and copy.
  In Generation mode we work with id's from vocabulary and in copy mode we work
  with positions in the sentence. In order to be able to handle these, we map
  positions to negetive integers with one shift them because of <UNK> token.
  So, <UNK> token is "0" in both mode, in generation for example "5" is the 5th
  word in vocab, and in copy, for exampl "-1" is the token at position "0" in
  the input entence and "-2" is token in position "1" in the input sentence.
"""


import numpy as np
import tensorflow as tf
import library



def get_bilinearterm_cop(config, enc_top_states, w_copy, decoder_outputs):
  """return the bilinear term (with tanh as nonlinearity) for copy mode.

  (according to the paper, https://arxiv.org/pdf/1603.06393.pdf, using the tanh
  non-linearity works better than linear transformation"
  Args:
    config: configurations
    enc_top_states: encode top state values
    [batch_size,(res_token+input_len), 2*num_hidden]
    w_copy: weight matrix for copy mode: [2*num_hidden, num_hidden]
    decoder_outputs: decoder output: [batch_size, *num_hidden]

  Returns:
     bilinear term
  """

  # h: from [batch_size,(res_token+input_len), 2*num_hidden]
  # to [(res_token+input_len), batch_size, 2*num_hidden]
  encoder_final_output = tf.transpose(enc_top_states, perm=[1, 0, 2])

  # h: from [(res_token+input_len), batch_size, 2*num_hidden]
  # to [(res_token+input_len) * batch_size, 2*num_hidden]
  encoder_final_output = tf.reshape(encoder_final_output, [(
      config.max_input_len + 1) * config.batch_size, 2 * config.num_hidden])

  # hw:  [(res_token+input_len) * batch_size, num_hidden]
  inc_out_w_copy = tf.tanh(tf.matmul(encoder_final_output, w_copy))  # tanh
  # inc_out_w_copy = tf.matmul(encoder_final_output,w_copy)

  # hw: from [(res_token+input_len) * batch_size, num_hidden]
  # to [(res_token+input_len), batch_size, num_hidden]
  enc_out_w_copy = tf.reshape(inc_out_w_copy, [(
      config.max_input_len + 1), config.batch_size, config.num_hidden])

  # s: from [batch_size, *num_hidden] to [1, batch_size, num_hidden]
  decoder_outputs_i = tf.expand_dims(decoder_outputs, 0)

  # s.hw (elementwise):
  # [1, batch_size, num_hidden].[(res_token+input_len), batch_size, num_hidden]
  dec_out_inc_out_w_copy = tf.multiply(enc_out_w_copy, decoder_outputs_i)

  # swh: sum ove num_hidden dimention -> [(res_token+input_len), batch_size]
  dec_out_inc_out_w_copy = tf.reduce_sum(dec_out_inc_out_w_copy, 2)

  return tf.transpose(dec_out_inc_out_w_copy)  # [batch_size, (1+input_len)]


class Model(object):
  """Wrapper for Tensorflow model graph for text sum vectors."""

  def __init__(self, config, mode, num_gpus=0):
    self._config = config
    self._mode = mode
    self._input_vocab = config.input_vocab
    self._output_vocab = config.output_vocab
    self._num_gpus = num_gpus
    self._cur_gpu = 0

  def run_train_step(self, sess, encoder_inputs, decoder_inputs, targets_gen,
                     targets_cop, encoder_len, decoder_len, loss_weights):

    to_return = [
        self._train_op, self._summaries, self._loss_gen, self.global_step
    ]
    return sess.run(
        to_return,
        feed_dict={
            self._encoder_inputs: encoder_inputs,
            self._decoder_inputs: decoder_inputs,
            self._targets_gen: targets_gen,
            self._targets_cop: targets_cop,
            self._encoder_len: encoder_len,
            self._decoder_len: decoder_len,
            self._loss_weights: loss_weights
        })

  def run_eval_step(self, sess, encoder_inputs, decoder_inputs, targets_gen,
                    targets_cop, encoder_len, decoder_len, loss_weights):
    to_return = [self._summaries, self._loss_gen, self.global_step]
    return sess.run(
        to_return,
        feed_dict={
            self._encoder_inputs: encoder_inputs,
            self._decoder_inputs: decoder_inputs,
            self._targets_gen: targets_gen,
            self._targets_cop: targets_cop,
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
    self._targets_gen = tf.placeholder(
        tf.int32, [config.batch_size, max_output_len], name='targets_gen')
    self._targets_cop = tf.placeholder(
        tf.int32, [config.batch_size, max_output_len], name='targets_cop')
    self._encoder_len = tf.placeholder(
        tf.int32, [config.batch_size], name='encoder_len')
    self._decoder_len = tf.placeholder(
        tf.int32, [config.batch_size], name='decoder_len')
    self._loss_weights = tf.placeholder(
        tf.float32, [config.batch_size, max_output_len], name='loss_weights')

  def _add_seq2seq(self):
    """Graph for the model."""
    config = self._config

    with tf.variable_scope('text2text'):
      encoder_inputs = tf.unstack(self._encoder_inputs, axis=1)
      decoder_inputs = tf.unstack(self._decoder_inputs, axis=1)
      targets_gen = tf.unstack(self._targets_gen, axis=1)
      targets_cop = tf.unstack(self._targets_cop, axis=1)
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
        emb_decoder_inputs = []

        emb_encoder_inputs_stacked = tf.stack(
            emb_encoder_inputs)  # [inp_len, batch_size, num_hidden]

        for i, _input in enumerate(decoder_inputs):
          positions = -1 - _input  # [batch_size]
          positions = tf.minimum(
              tf.maximum(positions, 0), config.min_input_len - 1)
          indecies = tf.minimum(
              tf.maximum(_input, 0), output_vsize - 1)
          emb_decoder_inputs.append(
              tf.where(
                  _input < 0,  # if copy
                  tf.transpose(
                      tf.matrix_diag_part(
                          tf.transpose(
                              tf.gather(emb_encoder_inputs_stacked, positions),
                              [2, 0, 1]))),  # indexing to [b_size, num_hidden]
                  tf.nn.embedding_lookup(output_embedding, indecies)))

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
        # weight matrix and bias for gneration mode logits layer
        w_generate = tf.get_variable(
            'w_gen', [config.num_hidden, output_vsize],
            dtype=tf.float32,
            initializer=tf.truncated_normal_initializer(stddev=1e-4))
        v_generate = tf.get_variable(
            'v_gen', [output_vsize],
            dtype=tf.float32,
            initializer=tf.truncated_normal_initializer(stddev=1e-4))

        # weight matrix for copy mode logits layer
        w_copy = tf.get_variable(
            'w_copy', [2 * config.num_hidden, config.num_hidden],
            dtype=tf.float32,
            initializer=tf.truncated_normal_initializer(stddev=1e-4))

        # weight matrix and bias for copy-generation switch gate
        w_switch = tf.get_variable(
            'w_switch', [config.num_hidden, 1],
            dtype=tf.float32,
            initializer=tf.truncated_normal_initializer(stddev=1e-4))
        v_switch = tf.get_variable(
            'v_switch', [1],
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
        self._enc_top_states = tf.concat(encoder_outputs, 1)  # 4 * 60 * 256
        self._dec_in_state = fw_state
        # Always use initial state attention because we need it for beam search
        initial_state_attention = True
        decoder_outputs, self._dec_out_state = (
            tf.contrib.legacy_seq2seq.attention_decoder(
                emb_decoder_inputs,
                self._dec_in_state,
                self._enc_top_states,
                cell,
                num_heads=config.num_heads,
                loop_function=None,
                initial_state_attention=initial_state_attention))

      with tf.variable_scope('output'), tf.device(self._next_device()):

        generator_outputs = []
        copier_outputs = []
        is_copy_switch_outputs = []

        for i, _output in enumerate(decoder_outputs):
          if i > 0:
            tf.get_variable_scope().reuse_variables()

          # generation mode output [batch_size,vocab_size)
          generator_outputs_i = tf.nn.xw_plus_b(_output, w_generate,
                                                v_generate)

          generator_outputs.append(generator_outputs_i)

          # copy mode output
          # add unk tokens to the encoder top state
          unk_token = tf.get_variable(
              'unk_token', [1, 1, 2 * self._config.num_hidden],
              dtype=tf.float32,
              initializer=tf.truncated_normal_initializer(stddev=1e-4))
          unk_token = tf.tile(unk_token, [config.batch_size, 1, 1])
          tmp_enc_top_states = tf.concat(
              [unk_token, self._enc_top_states], axis=1)

          copier_outputs.append(
              get_bilinearterm_cop(config, tmp_enc_top_states, w_copy,
                                   _output))

          # switch output
          is_copy_switch_outputs.append(
              tf.sigmoid(
                  tf.nn.xw_plus_b(_output, w_switch, v_switch)))

      # If decoding, use argmax to get output in words
      if self._mode == 'decode':
        with tf.variable_scope('decode_output'), tf.device('/cpu:0'):

          # generation mode
          best_outputs_gen = [tf.argmax(x, 1) for x in generator_outputs]
          tf.logging.info('best_outputs_gen%s',
                          best_outputs_gen[0].shape_as_list())
          self._outputs_gen = tf.concat(
              [tf.reshape(x, [config.batch_size, 1])
               for x in best_outputs_gen], 1)
          # copy mode
          best_outputs_cop = [tf.argmax(x, 1) for x in copier_outputs]
          tf.logging.info('best_outputs_cop%s',
                          best_outputs_cop[0].shape_as_list())
          self._outputs_cop = tf.concat(
              [tf.reshape(x, [config.batch_size, 1])
               for x in best_outputs_cop], 1)

          # switch gate
          self._outputs_is_copy_weights = is_copy_switch_outputs[-1]

          # generation mode
          topk_probs_gen, self._topk_ids_gen = tf.nn.top_k(
              tf.nn.softmax(generator_outputs[-1]), config.batch_size * 2)
          # copy mode
          topk_probs_cop, self._topk_ids_cop = tf.nn.top_k(
              tf.nn.softmax(copier_outputs[-1]), config.batch_size * 2)
          # switch gate
          self._topk_outputs_is_copy = tf.tile(self._outputs_is_copy_weights,
                                               [1, config.batch_size * 2])

          # normalize probs in gen and cop to make them comparable
          def prob_normalizer(a):
            return tf.nn.l2_normalize(a, 1)

          topk_norimalized_probs_cop = prob_normalizer(topk_probs_cop)
          topk_norimalized_probs_gen = prob_normalizer(topk_probs_gen)
          # weighting probabilitites based on switch gate output
          self._topk_log_probs_cop = tf.log(topk_norimalized_probs_cop *
                                            self._topk_outputs_is_copy)
          self._topk_log_probs_gen = tf.log(topk_norimalized_probs_gen *
                                            (1 - self._topk_outputs_is_copy))

      with tf.variable_scope('loss'), tf.device(self._next_device()):

        unk_token_id = 0

        def loss_func_cg(labels, inputs):
          with tf.device('/cpu:0'):  # Try gpu.
            labels = tf.reshape(labels, [-1])
            return tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=labels, logits=inputs)

        if config.num_softmax_samples != 0 and self._mode == 'train':
          tf.logging.info(
              'Sorry, Sampled Softmax is not available for copynet. '
              'I\'m going to use Softmax. Thank you for your understanding.')

        self._loss = library.sequence_loss_cg(
            generator_outputs, copier_outputs, targets_gen, targets_cop,
            is_copy_switch_outputs, loss_weights, loss_func_cg, loss_func_cg,
            config.switch_gate_ignore_prob,
            config.is_copy_when_switch_gate_ignored, unk_token_id,
            config.unk_token_penalty)

        self._loss_gen = tf.contrib.legacy_seq2seq.sequence_loss(
            generator_outputs, targets_gen, loss_weights)

        self._loss_cop = tf.contrib.legacy_seq2seq.sequence_loss(
            copier_outputs, targets_cop, loss_weights)

        tf.summary.scalar('loss/gen_min12', tf.minimum(12.0, self._loss_gen))
        tf.summary.scalar('loss/cop_min12', tf.minimum(12.0, self._loss_cop))
        tf.summary.scalar('loss/min12', tf.minimum(12.0, self._loss))
        tf.summary.scalar('switch_gate', tf.reduce_mean(is_copy_switch_outputs))

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
        tf.train.exponential_decay(config.lr, self.global_step,
                                   config.decay_steps, config.decay_rate))

    optimizer = tf.train.AdamOptimizer(lr_rate, epsilon=config.adam_epsilon)

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

  def decode_topk(self, sess, latest_tokens_or_position, enc_top_states,
                  dec_init_states, enc_inputs):

    # latest_tokens is going to be re-written later (in the loop function)
    """Return the topK results and new decoder states."""
    feed = {
        self._encoder_inputs:
            enc_inputs,
        self._enc_top_states:
            enc_top_states,
        self._dec_in_state:
            np.squeeze(np.array(dec_init_states)),
        self._decoder_inputs:
            np.transpose(np.array([latest_tokens_or_position])),
        self._decoder_len:
            np.ones([len(dec_init_states)], np.int32)
    }

    raw_results = sess.run(
        [
            self._topk_ids_gen, self._topk_log_probs_gen, self._topk_ids_cop,
            self._topk_log_probs_cop, self._topk_outputs_is_copy,
            self._dec_out_state
        ],
        feed_dict=feed)

    topk_ids_gen = raw_results[0]
    topk_log_probs_gen = raw_results[1]

    topk_ids_cop = raw_results[2]
    topk_log_probs_cop = raw_results[3]

    topk_is_copy = raw_results[4]
    states = raw_results[5]

    # integratge
    topk_ids = np.where(topk_is_copy > 0.5, -1 * topk_ids_cop, topk_ids_gen)
    topk_log_probs = np.where(topk_is_copy > 0.5, topk_log_probs_cop,
                              topk_log_probs_gen)

    new_states = [s for s in states]

    return (topk_ids, topk_log_probs, new_states)

  def build_graph(self):
    self._add_placeholders()
    self._add_seq2seq()
    self.global_step = tf.Variable(0, name='global_step', trainable=False)
    if self._mode == 'train':
      self._add_train_op()
    self._summaries = tf.summary.merge_all()
