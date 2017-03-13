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

"""Module for computing evaluation scores.
"""

import math


def count_ngram(token_c, token_r, n):
  """Count n-grams of length n."""
  clipped_count = 0
  count = 0
  r = 0
  c = 0

  # Calculate precision
  ref_counts = []
  ref_lengths = []

  # Build dictionary of ngram counts
  ngram_d = {}
  ref_lengths.append(len(token_r))
  limits = len(token_r) - n + 1
  # loop through the sentance consider the ngram length
  for i in range(limits):
    ngram = ' '.join(token_r[i:i + n]).lower()
    if ngram in ngram_d.keys():
      ngram_d[ngram] += 1
    else:
      ngram_d[ngram] = 1
  ref_counts.append(ngram_d)

  # candidate
  cand_dict = {}
  limits = len(token_c) - n + 1
  for i in range(0, limits):
    ngram = ' '.join(token_c[i:i + n]).lower()
    if ngram in cand_dict:
      cand_dict[ngram] += 1
    else:
      cand_dict[ngram] = 1
  clipped_count += clip_count(cand_dict, ref_counts)
  count += limits
  r += best_length_match(ref_lengths, len(token_c))
  c += len(token_c)

  if clipped_count == 0:
    pr = 0
  else:
    pr = float(clipped_count) / count
  bp = brevity_penalty(c, r)
  return pr, bp


def clip_count(cand_d, ref_ds):
  """Count the clip count for each ngram considering all references."""
  count = 0
  for m in cand_d.keys():
    m_w = cand_d[m]
    m_max = 0
    for ref in ref_ds:
      if m in ref:
        m_max = max(m_max, ref[m])
    m_w = min(m_w, m_max)
    count += m_w
  return count


def best_length_match(ref_l, cand_l):
  """Find the closest length of reference to that of candidate."""
  least_diff = abs(cand_l - ref_l[0])
  best = ref_l[0]
  for ref in ref_l:
    if abs(cand_l - ref) < least_diff:
      least_diff = abs(cand_l - ref)
      best = ref
  return best


def brevity_penalty(c, r):
  """Brevity penalty prevents short candidates from receiving high a score."""
  if c > r:
    bp = 1
  else:
    bp = math.exp(1 - (float(r) / c))
  return bp


def geometric_mean(precisions):
  """Compute geometric mean of list of variables."""
  prod = 1.0
  for prec in precisions:
    prod *= prec

  return prod**(1.0 / len(precisions))


def get_bleu(candidate, refs):
  """Compute bleu score for candidate and list of references."""

  prec = []
  brev = []

  # split into tokens
  token_c = candidate.strip().split()

  for ref in refs:
    token_r = ref.strip().split()

    if len(token_c) * len(token_r) == 0:
      continue

    for i in range(3):
      pr, bp = count_ngram(token_c, token_r, i + 1)
      prec.append(pr)
      brev.append(bp)

  if not prec:
    return 0.0

  score = geometric_mean(prec) * max(brev)
  return score


def get_f1(candidate, refs):
  """Compute F1 score for candidate and list of references."""

  # split into tokens & remove dublicates
  token_c = candidate.strip().split()
  token_c = set(token_c)

  score = 0.0

  if not token_c:
    return score

  for ref in refs:

    token_r = ref.strip().split()
    token_r = set(token_r)

    if not token_r:
      continue

    # compute true positives, precision & recall
    token_both = token_c.intersection(token_r)
    prec = len(token_both) / float(len(token_c))
    rec = len(token_both) / float(len(token_r))

    beta = 1.0
    if prec != 0 and rec != 0:
      score = max(score,
                  ((1 + beta**2) * prec * rec) / float(rec + beta**2 * prec))

  return score


def get_exact(candidate, refs):
  """Compute exact match score for candidate and list of references."""

  candidate = candidate.strip()

  for ref in refs:
    if candidate == ref.strip():
      return 1.0

  return 0.0
