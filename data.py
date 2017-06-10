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

"""Data batchers for data described in ..//data_prep/README.md."""

import glob
import random
import struct
import sys

from tensorflow.core.example import example_pb2

# Special tokens
SENTENCE_START = "<s>"
SENTENCE_END = "</s>"
UNKNOWN_TOKEN = "<UNK>"
PAD_TOKEN = "<PAD>"


class Vocab(object):
  """Vocabulary class for mapping words and ids."""

  def __init__(self, vocab_file, max_size):
    self._word_to_id = {}
    self._id_to_word = {}
    self._count = 0

    with open(vocab_file, 'r') as vocab_f:
      for line in vocab_f:
        pieces = line.split()
        if len(pieces) != 2:
          sys.stderr.write('Bad line: %s\n' % line)
          continue
        if pieces[0] in self._word_to_id:
          raise ValueError('Duplicated word: %s.' % pieces[0])
        self._word_to_id[pieces[0]] = self._count
        self._id_to_word[self._count] = pieces[0]
        self._count += 1
        if self._count > max_size:
          raise ValueError('Too many words: >%d.' % max_size)

  def CheckVocab(self, word):
    """Check if a word is in vocabulary"""
    if word not in self._word_to_id:
      return None
    return self._word_to_id[word]

  def WordToId(self, word):
    """Return Id of the word"""
    if word not in self._word_to_id:
      return self._word_to_id[UNKNOWN_TOKEN]
    return self._word_to_id[word]

  def IdToWord(self, word_id):
    """Return Word corresponding to an id in vocabulary"""
    if word_id not in self._id_to_word:
      raise ValueError('id not found in vocab: %d.' % word_id)
    return self._id_to_word[word_id]

  def NumIds(self):
    """Total Ids"""
    return self._count


def ExampleGen(data_path, num_epochs=None):
  """Generates tf.Examples from path of data files.

    Binary data format: <length><blob>. <length> represents the byte size
    of <blob>. <blob> is serialized tf.Example proto. The tf.Example contains
    the tokenized article text and summary.

  Args:
    data_path: path to tf.Example data files.
    num_epochs: Number of times to go through the data. None means infinite.

  Yields:
    Deserialized tf.Example.

  If there are multiple files specified, they accessed in a random order.
  """
  epoch = 0
  while True:
    if num_epochs is not None and epoch >= num_epochs:
      break
    filelist = glob.glob(data_path)
    assert filelist, 'Empty filelist.'
    random.shuffle(filelist)
    for f in filelist:
      reader = open(f, 'rb')
      while True:
        len_bytes = reader.read(8)
        if not len_bytes:
          break
        str_len = struct.unpack('q', len_bytes)[0]
        example_str = struct.unpack('%ds' % str_len, reader.read(str_len))[0]
        yield example_pb2.Example.FromString(example_str)

    epoch += 1


def Pad(ids, pad_id, length):
  """Pad or trim list to len length.

  Args:
    ids: list of ints to pad
    pad_id: what to pad with
    length: length to pad or trim to

  Returns:
    ids trimmed or padded with pad_id
  """
  assert pad_id is not None
  assert length is not None

  if len(ids) < length:
    a = [pad_id] * (length - len(ids))
    return ids + a
  else:
    return ids[:length]


def GetWordIds(text, vocab, pad_len=None, pad_id=None):
  """Get ids corresponding to words in text.

  Assumes tokens separated by space.

  Args:
    text: a string
    vocab: TextVocabularyFile object
    pad_len: int, length to pad to
    pad_id: int, word id for pad symbol

  Returns:
    A list of ints representing word ids.
  """
  ids = []
  for w in text.split():
    i = vocab.WordToId(w)
    if i >= 0:
      ids.append(i)
    else:
      ids.append(vocab.WordToId(UNKNOWN_TOKEN))
  if pad_len is not None:
    return Pad(ids, pad_id, pad_len)
  return ids


def GetWordIndices(
        target,
        source,
        vocab,
        position_based_indexing,
        pad_len=None,
        pad_id=None,):
  """Get index of any word of target in source.

  Assumes tokens separated by space.

  Args:
    target: target sentence
    source: source sentence
    vocab: TextVocabularyFile object
    position_based_indexing: boolean, true: in case we have multuple occurrence
    of the target in source, we take the nearest one based on position.
    pad_len: int, length to pad to
    pad_id: int, word id for pad symbol

  Returns:
    A list of ints representing word ids.
  """

  source_words = source.split()
  target_words = target.split()
  indices = []
  for j, w in enumerate(target_words):
    if w not in source_words:
      indices.append(vocab.WordToId(UNKNOWN_TOKEN))
    else:
      all_occurance = [i for i, token in enumerate(source_words) if token == w]
      if position_based_indexing:
        # consider the nearest occurance of source wrt the position in target
        # pylint: disable=cell-var-from-loop
        source_index = min(all_occurance, key=lambda x: abs(x - j))
        # pylint: enable=cell-var-from-loop
      else:
        # consider the first occurance
        source_index = all_occurance[0]
      indices.append(source_index + 1)  # shift positions

  if pad_len is not None:
    return Pad(indices, pad_id, pad_len)
  return indices


def Ids2Words(ids_list, vocab):
  """Get words from ids.

  Args:
    ids_list: list of int32
    vocab: TextVocabulary object

  Returns:
    List of words corresponding to ids.
  """
  assert isinstance(ids_list, list), "%s  is not a list" % ids_list
  return [vocab.IdToWord(i) for i in ids_list]


def Ids2Sentence(ids_list, vocab):
  """Get words from ids.

  Args:
    ids_list: list of int32
    vocab: TextVocabulary object

  Returns:
    Sentence with space separated words corresponding to ids.
  """

  return " ".join(Ids2Words(ids_list, vocab))


def WordIdBatcher(path, batch_size, vocab, src_key, dest_key, total_epochs=1):
  """Generates batch of word-ids from a vocabulary.

  Args:
    path: path to tf.Example files
    batch_size: int, batch size
    vocab: TextVocabularyFile object mapping words to ids.
    src_key: key of Example.Feature to use as source sequence
    dest_key: key of Example.Feature to use as destination sequence
    total_epochs: How many epochs before StopIteration

  Yields:
    src_ids: A list of length batch_size of word-id lists.
    dest_ids: A list of length batch_size of word-id lists.
  """
  ex_gen = ExampleGen(path)
  num_epochs = 0
  while True:
    src_batch = []
    dest_batch = []
    for _ in range(batch_size):
      try:
        e = ex_gen.next()
      except StopIteration:
        if num_epochs < total_epochs:
          # Reset batcher
          ex_gen = ExampleGen(path)
          src_batch, dest_batch = ex_gen.next()
        else:
          raise StopIteration("Hit total_epochs: %d" % (total_epochs))
      src_batch.append(GetWordIds(e.features.feature[src_key].bytes_list.value[
          0], vocab))
      dest_batch.append(GetWordIds(e.features.feature[
          dest_key].bytes_list.value[0], vocab))
      # This will throw an exception if src_key/dest_key are not found
    yield src_batch, dest_batch
