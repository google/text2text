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

"""Module for decoding.

It uses beam search algorithm to convert the model output (ids with probability)
to human readable text.
"""

import os
import time

import tensorflow as tf
import beam_search
import data
import metrics

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('max_decode_steps', 1000000,
                            'Number of decoding steps.')
tf.app.flags.DEFINE_integer('decode_batches_per_ckpt', 8000,
                            'Number of batches to decode before restoring next '
                            'checkpoint')

DECODE_START_DELAY_SECS = 0
DECODE_LOOP_DELAY_SECS = 60
DECODE_IO_FLUSH_INTERVAL = 100


class DecodeIO(object):
  """Writes the decoded and references to RKV files for Rouge score.

    See nlp/common/utils/internal/rkv_parser.py for detail about rkv file.
  """

  def __init__(self, outdir):
    """DecodeIO initializer.

    Args:
      outdir: Output directory for writing decode results.
    """
    self._write_count = 0
    self._outdir = outdir
    if not os.path.exists(self._outdir):
      os.makedirs(self._outdir)
    self._ref_file = None
    self._summary_file = None

  def Write(self, source, targets, decode, bleu, rouge, f1, exact):
    """Writes the target and decoded outputs to RKV files.

    Args:
      reference: The human (correct) result.
      decode: The machine-generated result
    """

    self._summary_file.write('source = %s\n' % source)
    for target in targets:
      self._summary_file.write('target = %s\n' % target)
    self._summary_file.write('decode = %s\n' % decode)
    self._summary_file.write(
        'bleu=%0.3f f1=%0.3f exact=%0.3f\n\n' % (bleu, f1, exact))
    self._write_count += 1
    if self._write_count % DECODE_IO_FLUSH_INTERVAL == 0:
        self._summary_file.flush()

  def ResetFiles(self):
    """Resets the output files. Must be called once before Write()."""
    if self._summary_file:
      self._summary_file.close()
    timestamp = time.strftime('%Y%m%d%H%M%S', time.gmtime())

    self._summary_file = open(
        os.path.join(self._outdir, 'summary%s' % timestamp), 'w')


class BeamSearch(object):
  """Beam search decoder.

    Example:
      decoder = BeamSearchDecoder(...)
      decoder.DecodeLoop()
  """

  def __init__(self, model, batch_reader, config):
    """Initializer of BeamSearchDecoder.

    Args:
      model: The neural network model.
      batch_reader: The batch data reader.
      config: Hyperparamters.
    """
    self._model = model
    self._model.build_graph()
    self._batch_reader = batch_reader
    self._config = config
    self._input_vocab = config.input_vocab
    self._output_vocab = config.output_vocab
    self._saver = tf.train.Saver()
    self._decode_io = DecodeIO(FLAGS.log_root + '/decode')

  def DecodeLoop(self):
    """Decoding loop for long running process.

      It's an infinite loop that repeatedly:
      1) Loads the latest checkpoint model,
      2) Reads sources from data files, generates source summaries with the
         checkpoint model, decode the source summaries.
      3) Write the decoded source summaries to file.
    """
    time.sleep(DECODE_START_DELAY_SECS)
    summary_writer = tf.summary.FileWriter(FLAGS.log_root + '/decode')
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    decode_step = 0
    old_global_step = -1
    while decode_step < self._config.max_decode_steps:
      time.sleep(DECODE_LOOP_DELAY_SECS)
      new_global_step = self._Decode(self._saver, summary_writer, sess,
                                     old_global_step)
      old_global_step = new_global_step
      decode_step += 1
      summary_writer.flush()

  def _Decode(self, saver, summary_writer, sess, old_global_step):
    """Restores a checkpoint and decodes it.

    Args:
      saver: Tensorflow checkpoint saver.
      summary_writer: Tensorflow summary writer.
      sess: Tensorflow session.
      old_global_step: Only decode of model is newer

    Returns:
      global_step: Step of model that was decodes.
    """
    checkpoint_state = tf.train.get_checkpoint_state(FLAGS.log_root)
    if not (checkpoint_state and checkpoint_state.model_checkpoint_path):
      tf.logging.info('No model to decode yet at %s', FLAGS.log_root)
      return old_global_step

    checkpoint_path = os.path.join(
        FLAGS.log_root,
        os.path.basename(checkpoint_state.model_checkpoint_path))

    saver.restore(sess, checkpoint_path)
    global_step = sess.run(self._model.global_step)
    if global_step <= old_global_step:
      tf.logging.info(
          'No new model to decode. Latest model at step %d (last: %d)',
          global_step, old_global_step)
      return old_global_step
    else:
      tf.logging.info('New model to decode. Loaded model at step %d (last: %d)',
                      global_step, old_global_step)

    scores = []
    self._decode_io.ResetFiles()
    for _ in range(self._config.decode_batches_per_ckpt):

      next_batch = self._batch_reader.NextBatch()
      if len(next_batch) == 9:  # normal batch reader
        (enc_inputs, _, _, enc_input_len, _, _, source, targets) = next_batch

      elif len(next_batch) == 10:  # copynet batch reader
        (enc_inputs, _, _, _, enc_input_len, _, _, source, targets) = next_batch
      else:
        tf.logging.error('Unknow batch reader is used... check the length of '
                         'return value of _batch_reader.NextBatch()')

      for i in range(self._config.batch_size):
        bs = beam_search.BeamSearch(
            self._model, self._config.batch_size,
            self._output_vocab.WordToId(data.SENTENCE_START),
            self._output_vocab.WordToId(data.SENTENCE_END),
            self._config.max_output_len)

        enc_inputs_copy = enc_inputs.copy()
        enc_inputs_copy[:] = enc_inputs[i:i + 1]
        enc_input_len_copy = enc_input_len.copy()
        enc_input_len_copy[:] = enc_input_len[i:i + 1]
        best_beam = bs.BeamSearch(sess, enc_inputs_copy, enc_input_len_copy)[0]
        dec_outputs = [int(t) for t in best_beam.tokens[1:]]
        score = self._DecodeBatch(source[i], targets[i], dec_outputs)
        scores.append(score)

    avg_score = [sum(x) / float(len(x)) for x in zip(*scores)]
    self._LogTensorboardSummary(summary_writer, 'metrics/bleu-3', avg_score[0],
                                global_step)
    self._LogTensorboardSummary(summary_writer, 'metrics/f1-measure',
                                avg_score[1], global_step)
    self._LogTensorboardSummary(summary_writer, 'metrics/exact', avg_score[2],
                                global_step)

    return global_step

  def _DecodeBatch(self, source, targets, dec_outputs):
    """Converts id to words and writes results.

    Args:
      source: The original source string.
      targets: The human (correct) target string.
      dec_outputs: The target word ids output by machine.

    Returns:
      List of metric scores for this batch.
    """
    output = ['None'] * len(dec_outputs)

    source_words = source.split()
    for i in range(len(dec_outputs)):
      if dec_outputs[i] < 0:  # it's from copier
        position = -1 - dec_outputs[i]
        if position < len(source_words):
          output[i] = source_words[position]
        else:
          output[i] = '<out_of_bound>'
      elif dec_outputs[i] >= 0:  # it's from generator or unk (if 0)
        output[i] = data.Ids2Words([dec_outputs[i]], self._output_vocab)[0]

    source = source.replace(data.SENTENCE_START + ' ', '').replace(
        ' ' + data.SENTENCE_END, '')
    targets = [
        x.replace(data.SENTENCE_START + ' ', '').replace(
            ' ' + data.SENTENCE_END, '') for x in targets
    ]
    decoded = ' '.join(output)
    end_p = decoded.find(data.SENTENCE_END, 0)
    if end_p != -1:
      decoded = decoded[:end_p].strip()

    bleu_score = metrics.get_bleu(decoded, targets)
    f1_score = metrics.get_f1(decoded, targets)
    exact_score = metrics.get_exact(decoded, targets)

    self._decode_io.Write(source, targets, decoded, bleu_score,
                          f1_score, exact_score)

    return bleu_score, f1_score, exact_score

  def _LogTensorboardSummary(self, summary_writer, variable_name, variable,
                             global_step):
    summary = tf.Summary()
    value = summary.value.add()
    value.tag = variable_name
    value.simple_value = variable
    summary_writer.add_summary(summary, global_step)
