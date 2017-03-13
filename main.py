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

"""Sequence to Sequence model for text.

"""
import importlib
import time

import tensorflow as tf
import data
import decode

FLAGS = tf.app.flags.FLAGS

# Where to find configuration
tf.app.flags.DEFINE_string('config', 'config/yoda.py', 'Path to configuration file')
tf.app.flags.DEFINE_string('override', '', 'CSV hyper-parameter override string.')

# Mode to run in
tf.app.flags.DEFINE_string('mode', 'train',
                  'train/eval/decode  mode.')

# Where to save output
tf.app.flags.DEFINE_string('log_root', 'log',
                    'Directory for model root.')
tf.app.flags.DEFINE_integer('export_version', 0, 'Export version number.')

# TF
tf.app.flags.DEFINE_integer('task', 0, 'Task id of the replica running the training.')
tf.app.flags.DEFINE_integer('num_gpus', 0, 'Number of gpus used.')

_RETRY_SLEEP_SECONDS = 10


def _RunningAvgLoss(loss, running_avg_loss, decay=0.999):
  """Calculates the running average of losses.

  Args:
    loss: loss of the single step.
    running_avg_loss: running average loss to be updated.
    decay: running average decay rate.

  Returns:
    Updated running_avg_loss.
  """
  if running_avg_loss == 0:
    running_avg_loss = loss
  else:
    running_avg_loss = running_avg_loss * decay + (1 - decay) * loss

  tf.logging.info('running_avg_loss: %f', running_avg_loss)
  return running_avg_loss


def _Train(model, config, data_batcher):
  """Runs text summarization model training loop.

    This is the entry function to start text summarization model training loop.

  Args:
    model: Model object.
    config: Config file with Hyperparameters
    data_batcher: Batcher object. It reads the source and target batches.

  Returns:
    running average loss.
  """
  with tf.device('/cpu:0'):
    model.build_graph()
    saver = tf.train.Saver(keep_checkpoint_every_n_hours=1.0)
    # Train dir is different from log_root to avoid summary directory
    # conflict with Supervisor.
    summary_writer = tf.summary.FileWriter(FLAGS.log_root + '/train')
    sv = tf.train.Supervisor(
        logdir=FLAGS.log_root,
        is_chief=(FLAGS.task == 0),
        saver=saver,
        summary_op=None,
        summary_writer=None,
        save_model_secs=config.checkpoint_secs,
        global_step=model.global_step)
    sess = sv.prepare_or_wait_for_session(
        config=tf.ConfigProto(allow_soft_placement=True))

    start_up_delay_steps = ((
        (FLAGS.task + 1) * FLAGS.task / 2) * config.start_up_delay_steps)

    while not sv.should_stop():
      global_steps = sess.run(model.global_step)
      tf.logging.info('step:%6d' % global_steps)
      if global_steps >= start_up_delay_steps:
        break
      time.sleep(_RETRY_SLEEP_SECONDS)

    tf.logging.info('Start training')
    running_avg_loss = 0
    global_steps = 0
    while not sv.should_stop() and global_steps < config.max_run_steps:
      next_batch = data_batcher.NextBatch()
      if len(next_batch) == 8:  # normal batch reader
        (encoder_inputs, decoder_inputs, targets, encoder_len, decoder_len,
         loss_weights, _, _) = next_batch
        (_, summaries, loss, global_steps) = model.run_train_step(
            sess, encoder_inputs, decoder_inputs, targets, encoder_len,
            decoder_len, loss_weights)
      elif len(next_batch) == 9:  # copynet batch reader
        (encoder_inputs, decoder_inputs, targets_gen, targets_cop, encoder_len,
         decoder_len, loss_weights, _, _) = next_batch
        (_, summaries, loss, global_steps) = model.run_train_step(
            sess, encoder_inputs, decoder_inputs, targets_gen, targets_cop,
            encoder_len, decoder_len, loss_weights)
      else:
        tf.logging.error('Unknow batch reader is used... check the length of '
                         'return value of _batch_reader.NextBatch()')

      summary_writer.add_summary(summaries, global_steps)
      running_avg_loss = _RunningAvgLoss(running_avg_loss, loss)
      global_steps += 1
      if global_steps % 100 == 0:
        summary_writer.flush()
    sv.Stop()
    return running_avg_loss


def _Eval(model, config, data_batcher):
  """Runs text summarization model evaluation loop.

    This is the entry function to start text summarization model evaluation
    loop.

  Args:
    model: Model object.
    config: Config file with Hyperparameters
    data_batcher: Batcher object. It reads the source and target batches.
  """
  model.build_graph()
  saver = tf.train.Saver()
  summary_writer = tf.summary.FileWriter(FLAGS.log_root + '/eval')
  sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
  running_avg_loss = 0
  best_model_loss = -1
  while True:
    time.sleep(config.eval_interval_secs)
    try:
      ckpt_state = tf.train.get_checkpoint_state(FLAGS.log_root)
    except tf.errors.OutOfRangeError as e:
      tf.logging.error('Cannot restore checkpoint: %s', e)
      continue

    if not (ckpt_state and ckpt_state.model_checkpoint_path):
      tf.logging.info('No model to eval yet at %s', FLAGS.log_root + '/train')
      continue

    tf.logging.info('Loading checkpoint %s', ckpt_state.model_checkpoint_path)
    saver.restore(sess, ckpt_state.model_checkpoint_path)

    next_batch = data_batcher.NextBatch()
    if len(next_batch) == 8:  # normal batch reader
      (encoder_inputs, decoder_inputs, targets, encoder_len, decoder_len,
       loss_weights, _, _) = next_batch
      (summaries, loss, global_steps) = model.run_eval_step(
          sess, encoder_inputs, decoder_inputs, targets, encoder_len,
          decoder_len, loss_weights)
    elif len(next_batch) == 9:  # copynet batch reader
      (encoder_inputs, decoder_inputs, targets_gen, targets_cop, encoder_len,
       decoder_len, loss_weights, _, _) = next_batch
      (summaries, loss, global_steps) = model.run_eval_step(
          sess, encoder_inputs, decoder_inputs, targets_gen, targets_cop,
          encoder_len, decoder_len, loss_weights)

    summary_writer.add_summary(summaries, global_steps)
    running_avg_loss = _RunningAvgLoss(running_avg_loss, loss)
    if global_steps % 100 == 0:
      summary_writer.flush()

    # Keep the model that reaches lowest loss on dev set.
    # was loaded, because we don't store the lowest loss in the model.
    if best_model_loss == -1 or best_model_loss > running_avg_loss:
      tf.logging.info('Found new best model (%f vs. %f)', running_avg_loss,
                      best_model_loss)
      best_model_loss = running_avg_loss
      # Don't write the state into the checkpoint proto to avoid automatic
      # age-related deletion.
      saver.save(sess, FLAGS.log_root + '/model.best', write_state=False)


def main(unused_argv):

  config = importlib.import_module('config.%s' % FLAGS.config)
  for argument in FLAGS.override.split(','):
    if '=' in argument:
      name = argument.split('=')[0]
      value = type(getattr(config, name))(argument.split('=')[1])
      setattr(config, name, value)
  config.input_vocab = data.Vocab(config.input_vocab_file,
                                   config.max_vocab_size)  # Max IDs
  if config.input_vocab.WordToId(data.PAD_TOKEN) <= 0:
    raise ValueError('Invalid PAD_TOKEN id.')
  # id of the UNKNOWN_TOKEN should be "0" for copynet model
  if config.input_vocab.WordToId(data.UNKNOWN_TOKEN) != 0:
    raise ValueError('Invalid UNKOWN_TOKEN id.')
  if config.input_vocab.WordToId(data.SENTENCE_START) <= 0:
    raise ValueError('Invalid SENTENCE_START id.')
  if config.input_vocab.WordToId(data.SENTENCE_END) <= 0:
    raise ValueError('Invalid SENTENCE_END id.')

  if config.output_vocab_file:
    config.output_vocab = data.Vocab(config.output_vocab_file,
                                     config.max_vocab_size)  # Max IDs
    if config.output_vocab.WordToId(data.PAD_TOKEN) <= 0:
      raise ValueError('Invalid PAD_TOKEN id.')
    # id of the UNKNOWN_TOKEN should be "0" for copynet model
    if config.output_vocab.WordToId(data.UNKNOWN_TOKEN) != 0:
      raise ValueError('Invalid UNKOWN_TOKEN id.')
    if config.output_vocab.WordToId(data.SENTENCE_START) <= 0:
      raise ValueError('Invalid SENTENCE_START id.')
    if config.output_vocab.WordToId(data.SENTENCE_END) <= 0:
      raise ValueError('Invalid SENTENCE_END id.')
  else:
    config.output_vocab = config.input_vocab

  train_batcher = config.Batcher(config.train_set, config)
  valid_batcher = config.Batcher(config.valid_set, config)
  tf.set_random_seed(config.random_seed)

  if FLAGS.mode == 'train':
    model = config.Model(config, 'train', num_gpus=FLAGS.num_gpus)
    _Train(model, config, train_batcher)
  elif FLAGS.mode == 'eval':
    config.dropout_rnn = 1.0
    config.dropout_emb = 1.0
    model = config.Model(config, 'eval', num_gpus=FLAGS.num_gpus)
    _Eval(model, config, valid_batcher)
  elif FLAGS.mode == 'decode':
    config.dropout_rnn = 1.0
    config.dropout_emb = 1.0
    config.batch_size = config.beam_size
    model = config.Model(config, 'decode', num_gpus=FLAGS.num_gpus)
    decoder = decode.BeamSearch(model, valid_batcher, config)
    decoder.DecodeLoop()

if __name__ == '__main__':
  tf.app.run()
