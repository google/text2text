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

"""Batch reader to sequence generation model, with bucketing support.

Continuously reads source and target from tf.Example file, batches the
sources and targets, and returns the batched inputs.

Example:
  batcher = batch_reader.Batcher(...)
  while True:
    (...) = batcher.NextBatch()
"""

import collections
import queue
from random import random
from random import shuffle
from threading import Thread
import time

import numpy as np
import tensorflow as tf

import data

ModelInput = collections.namedtuple(
    'ModelInput',
    'enc_input dec_input dec_target_g dec_target_c enc_len dec_len '
    'source targets')

BUCKET_CACHE_BATCH = 100
QUEUE_NUM_BATCH = 100
DAEMON_READER_THREADS = 16
BUCKETING_THREADS = 4


class Batcher(object):
  """Batch reader with shuffling and bucketing support."""

  def __init__(self, data_path, config):
    """Batcher initializer.

    Args:
      data_path: tf.Example filepattern.
      config: model hyperparameters.
    """
    self._data_path = data_path
    self._config = config
    self._input_vocab = config.input_vocab
    self._output_vocab = config.output_vocab
    self._source_key = config.source_key
    self._target_key = config.target_key
    self.use_bucketing = config.use_bucketing
    self._truncate_input = config.truncate_input
    self._input_queue = queue.Queue(QUEUE_NUM_BATCH * config.batch_size)
    self._bucket_input_queue = queue.Queue(QUEUE_NUM_BATCH)
    self._input_threads = []
    for _ in range(DAEMON_READER_THREADS):
      self._input_threads.append(Thread(target=self._FillInputQueue))
      self._input_threads[-1].daemon = True
      self._input_threads[-1].start()
    self._bucketing_threads = []
    for _ in range(BUCKETING_THREADS):
      self._bucketing_threads.append(Thread(target=self._FillBucketInputQueue))
      self._bucketing_threads[-1].daemon = True
      self._bucketing_threads[-1].start()

    self._watch_thread = Thread(target=self._WatchThreads)
    self._watch_thread.daemon = True
    self._watch_thread.start()

  def NextBatch(self):
    """Returns a batch of inputs for model.

    Returns:
      Tuple (enc_batch, dec_batch, target_gen_batch, target_cop_batch,
      enc_input_len, dec_input_len,
             loss_weights, origin_sources, origin_targets) where:
      enc_batch: A batch of encoder inputs [batch_size, config.enc_timestamps].
      dec_batch: A batch of decoder inputs [batch_size, config.dec_timestamps].
      target_gen_batch: A batch of targets [batch_size, config.dec_timestamps].
      target_cop_batch: A batch of targets [batch_size, config.dec_timestamps].
      enc_input_len: Encoder input lengths of the batch.
      dec_input_len: Decoder input lengths of the batch.
      loss_weights: Weights for loss function, 1 if not padded, 0 if padded.
      source: string. Original source words.
      targets: List of strings. Original target words.
    """
    enc_batch = np.zeros(
        (self._config.batch_size, self._config.max_input_len), dtype=np.int32)
    enc_input_lens = np.zeros((self._config.batch_size), dtype=np.int32)
    dec_batch = np.zeros(
        (self._config.batch_size, self._config.max_output_len), dtype=np.int32)
    dec_output_lens = np.zeros((self._config.batch_size), dtype=np.int32)
    target_gen_batch = np.zeros(
        (self._config.batch_size, self._config.max_output_len), dtype=np.int32)
    target_cop_batch = np.zeros(
        (self._config.batch_size, self._config.max_output_len), dtype=np.int32)

    loss_weights = np.zeros(
        (self._config.batch_size, self._config.max_output_len),
        dtype=np.float32)
    source = ['None'] * self._config.batch_size
    targets = [['None']] * self._config.batch_size

    buckets = self._bucket_input_queue.get()
    for i in range(self._config.batch_size):
      (enc_inputs, dec_inputs, dec_targets_gen, dec_targets_cop, enc_input_len,
       dec_output_len, source_i, targets_i) = buckets[i]

      enc_input_lens[i] = enc_input_len
      dec_output_lens[i] = dec_output_len
      enc_batch[i, :] = enc_inputs[:]
      dec_batch[i, :] = dec_inputs[:]
      target_gen_batch[i, :] = dec_targets_gen[:]
      target_cop_batch[i, :] = dec_targets_cop[:]
      source[i] = source_i
      targets[i] = targets_i
      for j in range(dec_output_len):
        loss_weights[i][j] = 1

    return (enc_batch, dec_batch, target_gen_batch, target_cop_batch,
            enc_input_lens, dec_output_lens, loss_weights, source, targets)

  def _FillInputQueue(self):
    """Fills input queue with ModelInput."""

    # input gets padded
    pad_id = self._input_vocab.WordToId(data.PAD_TOKEN)
    # output get start id and padded with end ids
    end_id = self._output_vocab.WordToId(data.SENTENCE_END)

    input_gen = self._TextGenerator(data.ExampleGen(self._data_path))
    while True:
      (source, targets) = next(input_gen)
      # target = choice(targets)
      target = targets[0]

      # Convert sentences to word IDs, stripping existing <s> and </s>.
      enc_inputs = data.GetWordIds(source, self._input_vocab)
      dec_inputs_gen = data.GetWordIds(target, self._output_vocab)
      dec_inputs_cop = data.GetWordIndices(
          target, source, self._input_vocab, position_based_indexing=True)

      # Filter out too-short input
      if len(enc_inputs) < self._config.min_input_len:
        tf.logging.warning('Drop an example - input to short: %d (min: %d)',
                           len(enc_inputs), self._config.min_input_len)
        continue

      if len(dec_inputs_gen) < self._config.min_input_len:
        tf.logging.warning('Drop an example - output to short: %d (min: %d)',
                           len(enc_inputs), self._config.min_input_len)
        continue

      # If we're not truncating input, throw out too-long input
      if not self._truncate_input:
        if len(enc_inputs) > self._config.max_input_len:
          tf.logging.warning('Drop an example - input to long: %d (max: %d)',
                             len(enc_inputs), self._config.max_input_len)
          continue
        if len(dec_inputs_gen) > self._config.max_output_len:
          tf.logging.warning('Drop an example - output to long: %d (max: %d)',
                             len(dec_inputs_gen), self._config.max_output_len)
          continue
      # If we are truncating input, do so if necessary
      else:
        if len(enc_inputs) > self._config.max_input_len:
          enc_inputs = enc_inputs[:self._config.max_input_len]
          dec_inputs_cop = [
              pos if pos <= self._config.max_input_len else 0
              for pos in dec_inputs_cop
          ]
        if len(dec_inputs_gen) > self._config.max_output_len:
          dec_inputs_gen = dec_inputs_gen[:self._config.max_output_len]
          dec_inputs_cop = dec_inputs_cop[:self._config.max_output_len]

      # dec_targets_gen is dec_inputs without <s> at beginning, plus </s> at end
      dec_targets_gen = dec_inputs_gen[1:]
      dec_targets_gen.append(end_id)

      # dec_targets_gen is dec_inputs without <s> at beginning, plus </s> at end
      dec_targets_cop = dec_inputs_cop[1:]
      end_position = len(enc_inputs)
      dec_targets_cop.append(end_position)

      enc_input_len = len(enc_inputs)
      dec_output_len = len(dec_targets_gen)  # is equal to len(dec_targets_cop)

      # Pad if necessary
      while len(enc_inputs) < self._config.max_input_len:
        enc_inputs.append(pad_id)
      while len(dec_inputs_gen) < self._config.max_output_len:
        dec_inputs_gen.append(end_id)
      while len(dec_targets_gen) < self._config.max_output_len:
        dec_targets_gen.append(end_id)
      while len(dec_targets_cop) < self._config.max_output_len:
        dec_targets_cop.append(end_position)

      element = ModelInput(enc_inputs, dec_inputs_gen, dec_targets_gen,
                           dec_targets_cop, enc_input_len, dec_output_len,
                           source, targets)
      self._input_queue.put(element)

  def _FillBucketInputQueue(self):
    """Fills bucketed batches into the bucket_input_queue."""
    while True:
      inputs = []
      for _ in range(self._config.batch_size * BUCKET_CACHE_BATCH):
        inputs.append(self._input_queue.get())
      if self.use_bucketing:
        inputs = sorted(inputs, key=lambda inp: inp.enc_len)

      batches = []
      for i in range(0, len(inputs), self._config.batch_size):
        batches.append(inputs[i:i + self._config.batch_size])
      shuffle(batches)
      for b in batches:
        self._bucket_input_queue.put(b)

  def _WatchThreads(self):
    """Watches the daemon input threads and restarts if dead."""
    while True:
      time.sleep(60)
      input_threads = []
      for t in self._input_threads:
        if t.is_alive():
          input_threads.append(t)
        else:
          tf.logging.error('Found input thread dead.')
          new_t = Thread(target=self._FillInputQueue)
          input_threads.append(new_t)
          input_threads[-1].daemon = True
          input_threads[-1].start()
      self._input_threads = input_threads

      bucketing_threads = []
      for t in self._bucketing_threads:
        if t.is_alive():
          bucketing_threads.append(t)
        else:
          tf.logging.error('Found bucketing thread dead.')
          new_t = Thread(target=self._FillBucketInputQueue)
          bucketing_threads.append(new_t)
          bucketing_threads[-1].daemon = True
          bucketing_threads[-1].start()
      self._bucketing_threads = bucketing_threads

  def _TextGenerator(self, example_gen):
    """Generates source and target text from tf.Example.

    Args:
      example_gen: ExampleGen that yields tf.Example.

    Yields:
      Tuple (source_text, target_text) where:
      source_text: Text source string.
      target_texts: Text targets (well-formed) string.
    """
    while True:
      example = next(example_gen)
      try:

        # TARGET
        all_target_texts = []
        if len(self._target_key.split(',')) > 1:
          all_target_text = ''
          counter = -1
          # concat different keys (not combinable with multiple targets)
          for key in self._target_key.split(','):
            if counter >= 0:
              all_target_text += ' '
            all_target_text += self._GetExFeatureText(example, key)[0].strip()
            counter += 1
          all_target_text = self._AddSentenceBoundary(all_target_text)
          all_target_texts.append(all_target_text)
        else:
          key = self._target_key
          for target_text in self._GetExFeatureText(example, key):
            target_text = target_text.strip()
            target_text = self._AddSentenceBoundary(target_text)
            all_target_texts.append(target_text)

        # SOURCE
        all_source_text = ''
        counter = -1
        # if input is list of keys we concat them using separator tokens.
        for key in self._source_key.split(','):
          if counter >= 0:
            # <sep_0>, etc. must already be part of the vocab
            if self._input_vocab.WordToId('<sep_' + str(counter) + '>') <= 0:
              tf.logging.error('Separator token missing: <sep_%s>',
                               str(counter))
            all_source_text += ' <sep_' + str(counter) + '> '
          # sepcial key to add the length of the output to the input
          if key == '%LENGTH%':
            all_source_text += str(len(all_target_texts[0].split()))
          elif len(key.split('%')) == 2:
            if random() < float(key.split('%')[0]) / 100:
              all_source_text += self._GetExFeatureText(
                  example, key.split('%')[1])[0].strip()
            else:
              all_source_text += ' <no_callout> '
          else:
            all_source_text += self._GetExFeatureText(example, key)[0].strip()
          counter += 1
        all_source_text = self._AddSentenceBoundary(all_source_text)

        yield (all_source_text, all_target_texts)

      except ValueError as e:
        tf.logging.error(e)
        tf.logging.error('Failed to get article or abstract from example')
        continue

  def _AddSentenceBoundary(self, text):
    """Pads text with start end end of sentence token iff needed.

    Args:
      text: text to be padded.

    Returns:
      A text with start and end tokens.
    """

    if not text.startswith(data.SENTENCE_START):
      text = data.SENTENCE_START + ' ' + text
    if not text.endswith(data.SENTENCE_END):
      text = text + ' ' + data.SENTENCE_END

    return text

  def _GetExFeatureText(self, example, key):
    """Extracts text for a feature from tf.Example.

    Args:
      example: tf.Example.
      key: Key of the feature to be extracted.

    Returns:
      A feature text extracted.
    """

    values = []
    for value in example.features.feature[key].bytes_list.value:
      values.append(value.decode("utf-8"))

    return values
