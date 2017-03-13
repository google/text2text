<h3>text2text: Implementation of variants of Sequence to Sequence model:</h3>

Authors:

* Sascha Rothe (rothe@google.com ),
* Mostafa Dehghani (github:mostafadehghani)

<b>Introduction</b>

The code contains different implementations of sequence to sequence models:

* Original Sequence to Sequence model with attention mechanism:
 [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473)
* Bag of Words to Sequence model:
Inspired by [Order Matters: Sequence to sequence for sets](https://arxiv.org/abs/1511.06391)
* Incorporating copy mechanism with Sequence to Sequence model:
Inspired by [Incorporating Copying Mechanism in Sequence-to-Sequence Learning](https://arxiv.org/abs/1603.06393)



<b>DataSet</b>


To prepare the dataset, see `ExampleGen` in `data.py` about the data format.
`data/data` contains a toy example. Also see `data/vocab`
for example vocabulary format. 
`data/data_convert_example.py` contains example of convert between binary and text.

<b>How To Run</b>

Pre-requesite:

Install TensorFlow and Bazel.

```shell
# cd to your workspace
# 1. Clone the text2text code to your workspace 'text2text' directory.
# 2. Create an empty 'WORKSPACE' file in your workspace.
# 3. Preapre the config file wrt the model you wish to run and put it in the
#    config directory.

ls -R
.:
text2text  WORKSPACE

./text2text:
batch_reader  beam_search.py  BUILD  config  data  data.py  decode.py  
__init__.py  library.py  main.py  metrics.py  model  README.md

./text2text/batch_reader:
copynet_batcher.py  __init__.py  vocab_batcher.py

./text2text/config:
cfg_copynet.py  cfg_seq2seq.py  cfg_bow2seq.py
 __init__.py 

./text2text/model:
copynet.py  __init__.py  seq2seq.py  bow2seq.py

./text2text/data:
data  data_convert_example.py  text_data  vocab


bazel build -c opt --copt=-mavx --config=cuda text2text:main

# Run the training.
bazel-bin/text2text/main \
    --mode=train \
    --config="cfg_seq2seq" \
    --log_root="text2text/log_root" \
    --override="eval_interval_secs=0" \
    --logtostderr

# Run the eval. Try to avoid running on the same machine as training.
bazel-bin/text2text/main \
    --mode=eval \
    --config="cfg_seq2seq" \
    --log_root="text2text/log_root" \
    --logtostderr

# Run the decode. Run it when the model is mostly converged.
bazel-bin/text2text/main \
  --mode=decode \
    --config="cfg_seq2seq" \
    --log_root="text2text/log_root" \
    --logtostderr
```

`--config="config_file_name"` determines the config file from the config dir 
 in which the model you wish to run, paths to data, and 
hyperparameters of the model are specified. There are sample config files for 
each models in config directory. The output of the code and summaries will be 
written to a `text2text/config_file_name` directory.
