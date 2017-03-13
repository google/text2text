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

"""Beam search module.

Beam search takes the top K results from the model, predicts the K results for
each of the previous K result, getting K*K results. Pick the top K results from
K*K results, and start over again until certain number of results are fully
decoded.
"""

import numpy as np
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_bool('normalize_by_length', True,
                  'Whether normalize probability by length for decode outputs.')


class Hypothesis(object):
  """Defines a hypothesis during beam search.

  Attributes:
    tokens: list of ints. Token ids produced by the model.
    log_probs: list of floats. log probs of the `tokens`.
    state: Decoder state used for generating next token.
  """

  def __init__(self, tokens, log_prob, state):
    """Hypothesis initializer.

    Args:
      tokens: list of ints. Start tokens for decoding.
      log_prob: list of floats. Log prob of the start tokens, usually 1.
      state: Decoder initial states.
    """
    self.tokens = tokens
    self.log_prob = log_prob
    self.state = state

  def Extend(self, token, log_prob, new_state):
    """Extends the hypothesis with result from latest step.

    Args:
      token: int. Latest token from decoding.
      log_prob: float. Log prob of the latest decoded tokens.
      new_state: Decoder output state. Fed to the decoder for next step.

    Returns:
      New Hypothesis with the results from latest step.
    """
    return Hypothesis(self.tokens + [token], self.log_prob + log_prob,
                      new_state)

  @property
  def latest_token(self):
    return self.tokens[-1]

  def __str__(self):
    return ('Hypothesis(log prob = %.4f, prop = %.4f, tokens = %s)' %
            (self.log_prob, np.exp(self.log_prob), self.tokens))


class BeamSearch(object):
  """Beam search."""

  def __init__(self, model, beam_size, start_token, end_token, max_steps):
    """BeamSearch initializer.

    Args:
      model: Model
      beam_size: int.
      start_token: int, Id of the token to start decoding with.
      end_token: int, Id of the token that completes an hypothesis.
      max_steps: int, Upper limit on the size of the hypothesis.
    """
    self._model = model
    self._beam_size = beam_size
    self._start_token = start_token
    self._end_token = end_token
    self._max_steps = max_steps

  def BeamSearch(self, session, enc_inputs, enc_sequence_len):
    """Performs beam search for decoding.

    Args:
      session: tf.Session.
      enc_inputs: ndarray of shape (enc_length, 1), the document ids to encode.
      enc_sequence_len: ndarray of shape (1), the length of the sequence.

    Returns:
      The best Hypotheses found by beam search, ordered by score.
    """

    # Run the encoder and extract the outputs and final state.
    enc_top_states, dec_in_state = self._model.encode_top_state(
        session, enc_inputs, enc_sequence_len)
    # Replicate the initial states K times for the first step.
    hypothesis = [Hypothesis([self._start_token], 0.0, dec_in_state)
                 ] * self._beam_size
    results = []

    steps = 0
    while steps < self._max_steps and len(results) < self._beam_size:
      latest_tokens = [h.latest_token for h in hypothesis]
      states = [h.state for h in hypothesis]
      topk_ids, topk_log_probs, new_states = self._model.decode_topk(
          session, latest_tokens, enc_top_states, states, enc_inputs)
      # Extend each hypothesis.
      all_hypothesis = []
      # The first step takes the best K results from first hypothesis. Following
      # steps take the best K results from K*K hypothesis.
      num_beam_source = 1 if steps == 0 else len(hypothesis)
      for i in range(num_beam_source):
        h, ns = hypothesis[i], new_states[i]
        for j in range(self._beam_size * 2):
          all_hypothesis.append(
              h.Extend(topk_ids[i, j], topk_log_probs[i, j], ns))

      # Filter and collect any hypotheses that have the end token.
      hypothesis = []
      for h in self._BestHypothesis(all_hypothesis):
        if h.latest_token == self._end_token:
          # Pull the hypothesis off the beam if the end token is reached.
          results.append(h)
        else:
          # Otherwise continue to the extend the hypothesis.
          hypothesis.append(h)
        if (len(hypothesis) == self._beam_size or
            len(results) == self._beam_size):
          break

      steps += 1

    if steps == self._max_steps:
      results.extend(hypothesis)

    return self._BestHypothesis(results)

  def _BestHypothesis(self, hypothesis):
    """Sorts the hypothesis based on log probs and length.

    Args:
      hypothesis: A list of Hypothesis objects.

    Returns:
      A list of sorted Hypothesis objects in reverse log_prob order.
    """
    # This length normalization is only effective for the final results.
    if FLAGS.normalize_by_length:
      return sorted(
          hypothesis, key=lambda h: h.log_prob / len(h.tokens), reverse=True)
    else:
      return sorted(
          hypothesis, key=lambda h: h.log_prob, reverse=True)
