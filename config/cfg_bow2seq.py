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

"""Batcher, Model and Hyperparameters for one run.

"""

source_key = 'source'
target_key = 'target'
train_set = 'data/data'
valid_set = 'data/data'
input_vocab_file = 'data/vocab'
output_vocab_file = 'data/vocab'

# Model settings
max_run_steps = 500000  # Maximum number of run steps.
beam_size = 4  # beam size for beam search decoding.

# Training settings
eval_interval_secs = 60  # How often to run eval.
checkpoint_secs = 60  # How often to checkpoint.
use_bucketing = False  # Whether bucket sources of similar length.
truncate_input = False  # Truncate inputs that are too long. Else discard.
max_decode_steps = 1000000  # Number of decoding steps
decode_batches_per_ckpt = 100  # batches to decode before loading next checkp.
start_up_delay_steps = 200  # i-th replica starts training after
# i*(i+1)/2*start_up_delay_steps steps

# Misc settings
random_seed = 111  # A seed value for randomness.

# Hyperparameter
max_vocab_size = 32000
batch_size = 64
enc_layers = 15  # stacked LSTM or number of iterations of the processor
max_input_len = 15
max_output_len = 15
min_input_len = 1  # discard sources/summaries < than this
num_hidden = 256  # for rnn cell
emb_dim = 128  # If 0 don't use embedding
num_heads = 1
max_grad_norm = 2
num_softmax_samples = 4096  # If 0 don't use sample softmax.
dropout_rnn = 1.0
dropout_emb = 1.0

# Optimizer settings
optimizer = 'adam'  # gradient_descent adam
adam_epsilon = 0.0001
min_lr = 0.000001  # min learning rate
lr = 0.001  # learning rate
decay_steps = 30000
decay_rate = 0.98
