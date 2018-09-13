This repository contains code for the ACL 2017 paper *[Get To The Point: Summarization with Pointer-Generator Networks](https://arxiv.org/abs/1704.04368)*. This version has been modified to tackle the paraphrasing problem using our Ensemble Paraphrase dataset.

## Accessing the Model
Currently, the data and most recent models are located on the ec2 V100 instance and can be ssh'd into as follows:

```
ssh -i ~/path/to/rsa-private-key ubuntu@35.153.162.101
```

My private key is stored at `~/.ssh/mm_admin_ec2` and looks as follows:

```
-----BEGIN RSA PRIVATE KEY-----
MIIEpQIBAAKCAQEAyy+1Xnr02Kw4Hsr0H4IrM6MYcSsFArxuxUSCViWqFCCAX4Fr
FIMCMH9S5s5b1XvdUBDdKx7sI2knarzYWnlSYgfmc49Mn3NL2q50bxWam1JL802j
MyPY3tG/6GPBnXWt9HCdKwMYRiuWdfr5s/ZZWQdz2ajOIxXIMO3c56PWP3+ReckT
Ee1gLiHXEeEpCUluhnNzvkYCcPQhOq1kpKX5o5w04Pd7YMJB/0+SaVTH1hj8ITzX
/Ite/ATjQfghCnZEdcdrlsMFXWC5mdSoR1r/AaIxtXHg4V9RRZRzbdQuVOsZgEdW
1nxKmymkyHDxLvgv3crzjfdR4p2GXvs3h9eAawIDAQABAoIBAQCripHTPHeu3mdb
B/nnCBy1McMW+Dz4vGHAxkBLvuzCpVX8S+U9ogWdMEGrfxllOXf1eude2nJ+yu5u
XRVD/R2iigz0D02faNXnwFWo21J2Q5q/xRI9wMv+ApPE/UOfTpUYxGLMx4Yc5/vV
i5h5aUixs2sJnky6rk8fkBJqtpXRcL8RQ1BMOgvuoE7QFatbqsaS6yG61nHnZN7W
8gdqoRs1tl5TBdDvcVUOA7ci9UqT+YykXYyu5xatcIcv3od/pdbx+pueUiHX+saZ
ajzz8wc5OotUZX3xfJ6GOwxTybLNXQS1+NrHTKzqkfY0LtuXuOXL/MqtYX3ZoRFg
twV4R7TBAoGBAOilNrYXWVCQBIux63EIN8Qs9Ptojzzz29PWXUBoLI6oUXoQ6w2j
m7FSOAMBoGDByiWQ7JtUYh6v8OPZwkrlZKhy86B9u0qGl/kqr3UUiwKibiqSSSwV
AUTcJNECbuIuFHD2L+/BT8c/qk4Tk1VqVKPp8wycols85Hq3VerbQq0bAoGBAN+V
bHGwwHV6Ug03gEGsL3X3b3Bzb4P5YEQoWuuXksFCEw0D1LQwkbW9XZE+Uh0Gwpjx
0UEuBjp0qv6ZpOj3OstYCUfleHADnWfBnlY2j2sSNvCvD1c0A5XlwCDwkE4qAQoT
o3O0MJMOEZjXUi+cFmK+BZo4dpqPVY5FHwBo0j7xAoGBANBkldKcraulpfzAXic8
9j8rCjNd4Nj3k9V48sUry6XqDjnqh7MK9/7pqDTBMzkYvgvSeq9U8iOM9JoVSJ4w
R29AsQz1M1y5nc5bNxoIUdEFUOufaB4tnpphHDWBGXRPVI4vaJc6MU1IsTQJAvQD
MFg3yzhj4mcZ74whR7v9tgc7AoGAGzCuyXy3SxIfHGBdhp8L/oqi0MgZDbGsFV3b
AL+kJ0CD1JdcrLUjYziuQlWT4tfJnpP54LVaKI27pc7FkVdPnR8NseKkLOsjn1Sa
jt1/v4OtjsObV/2JWhdgsCDRS9Ar5C/TNuT460lQtoepBQU0a7m0MLchJVa9Dto1
FPWqa+ECgYEAkftvanZUhK0DwKJzaD5kTcJyeP6fSx4eDHd24928CteLHLMzucaz
ZeUE/7qodeZkuSYvGEZDwvyBC8le4xgR2EvrBwskTjbxHEu4uf7BB2N9T8opxRgM
SEPo8upWuYchTJ/SCd6sT2/6CSJlnSttZbxTJ2fxrk3gzPM8V7GH+BY=
-----END RSA PRIVATE KEY-----
```
Once on the instance, run the following:
```
cd paraphrasing/pointer-generator # move into the repo
source activate tensorflow_p36 # activate the optimized tensorflow environment using python 3.6
```
## Screen versus running in background
For whatever reason, I noticed that running in background produces weird stopping behavior. If the connection to the server breaks, for instance, often the training will also break. It's possible the behavior I noticed was do to something else and feel free to run in background with `$ <command> &` but my recommendation is to use `screen`.

### Starting a screen
To start a screen simply run `screen`.

From here you must reactivate the tensorflow environment with `source activate tensorflow_p36`.

Now, you may run any commands that you would run normally and can furthermore run commands in the background (for instance if you simply want to write the output to file).

### Exiting from screen
There are two ways to exit from a running `screen`. The first way, and primary way, is to ~detach~ from the screen by simply pressing Control-A-D 

## Looking for test set output?
The test set output of the models described in the paper can be found [here](https://drive.google.com/file/d/0B7pQmm-OfDv7MEtMVU5sOHc5LTg/view?usp=sharing).

## Looking for pretrained model?
A pretrained model is available here:
* [Version for Tensorflow 1.0](https://drive.google.com/file/d/0B7pQmm-OfDv7SHFadHR4RllfR1E/view?usp=sharing)
* [Version for Tensorflow 1.2.1](https://drive.google.com/file/d/0B7pQmm-OfDv7ZUhHZm9ZWEZidDg/view?usp=sharing)

(The only difference between these two is the naming of some of the variables in the checkpoint. Tensorflow 1.0 uses `lstm_cell/biases` and `lstm_cell/weights` whereas Tensorflow 1.2.1 uses `lstm_cell/bias` and `lstm_cell/kernel`).

## Looking for CNN / Daily Mail data?
Instructions are [here](https://github.com/abisee/cnn-dailymail).

## About this code
This code is based on the [TextSum code](https://github.com/tensorflow/models/tree/master/textsum) from Google Brain.

This code was developed for Tensorflow 0.12, but has been updated to run with Tensorflow 1.0.
In particular, the code in attention_decoder.py is based on [tf.contrib.legacy_seq2seq_attention_decoder](https://www.tensorflow.org/api_docs/python/tf/contrib/legacy_seq2seq/attention_decoder), which is now outdated.
Tensorflow 1.0's [new seq2seq library](https://www.tensorflow.org/api_guides/python/contrib.seq2seq#Attention) probably provides a way to do this (as well as beam search) more elegantly and efficiently in the future.

## How to run

### Get the dataset
To obtain the CNN / Daily Mail dataset, follow the instructions [here](https://github.com/abisee/cnn-dailymail). Once finished, you should have [chunked](https://github.com/abisee/cnn-dailymail/issues/3) datafiles `train_000.bin`, ..., `train_287.bin`, `val_000.bin`, ..., `val_013.bin`, `test_000.bin`, ..., `test_011.bin` (each contains 1000 examples) and a vocabulary file `vocab`.

**Note**: If you did this before 7th May 2017, follow the instructions [here](https://github.com/abisee/cnn-dailymail/issues/2) to correct a bug in the process.

### Run training
To train your model, run:

```
python run_summarization.py --mode=train --data_path=/path/to/chunked/train_* --vocab_path=/path/to/vocab --log_root=/path/to/a/log/directory --exp_name=myexperiment
```

This will create a subdirectory of your specified `log_root` called `myexperiment` where all checkpoints and other data will be saved. Then the model will start training using the `train_*.bin` files as training data.

**Warning**: Using default settings as in the above command, both initializing the model and running training iterations will probably be quite slow. To make things faster, try setting the following flags (especially `max_enc_steps` and `max_dec_steps`) to something smaller than the defaults specified in `run_summarization.py`: `hidden_dim`, `emb_dim`, `batch_size`, `max_enc_steps`, `max_dec_steps`, `vocab_size`. 

**Increasing sequence length during training**: Note that to obtain the results described in the paper, we increase the values of `max_enc_steps` and `max_dec_steps` in stages throughout training (mostly so we can perform quicker iterations during early stages of training). If you wish to do the same, start with small values of `max_enc_steps` and `max_dec_steps`, then interrupt and restart the job with larger values when you want to increase them.

### Run (concurrent) eval
You may want to run a concurrent evaluation job, that runs your model on the validation set and logs the loss. To do this, run:

```
python run_summarization.py --mode=eval --data_path=/path/to/chunked/val_* --vocab_path=/path/to/vocab --log_root=/path/to/a/log/directory --exp_name=myexperiment
```

Note: you want to run the above command using the same settings you entered for your training job.

**Restoring snapshots**: The eval job saves a snapshot of the model that scored the lowest loss on the validation data so far. You may want to restore one of these "best models", e.g. if your training job has overfit, or if the training checkpoint has become corrupted by NaN values. To do this, run your train command plus the `--restore_best_model=1` flag. This will copy the best model in the eval directory to the train directory. Then run the usual train command again.

### Run beam search decoding
To run beam search decoding:

```
python run_summarization.py --mode=decode --data_path=/path/to/chunked/val_* --vocab_path=/path/to/vocab --log_root=/path/to/a/log/directory --exp_name=myexperiment
```

Note: you want to run the above command using the same settings you entered for your training job (plus any decode mode specific flags like `beam_size`).

This will repeatedly load random examples from your specified datafile and generate a summary using beam search. The results will be printed to screen.

**Visualize your output**: Additionally, the decode job produces a file called `attn_vis_data.json`. This file provides the data necessary for an in-browser visualization tool that allows you to view the attention distributions projected onto the text. To use the visualizer, follow the instructions [here](https://github.com/abisee/attn_vis).

If you want to run evaluation on the entire validation or test set and get ROUGE scores, set the flag `single_pass=1`. This will go through the entire dataset in order, writing the generated summaries to file, and then run evaluation using [pyrouge](https://pypi.python.org/pypi/pyrouge). (Note this will *not* produce the `attn_vis_data.json` files for the attention visualizer).

### Evaluate with ROUGE
`decode.py` uses the Python package [`pyrouge`](https://pypi.python.org/pypi/pyrouge) to run ROUGE evaluation. `pyrouge` provides an easier-to-use interface for the official Perl ROUGE package, which you must install for `pyrouge` to work. Here are some useful instructions on how to do this:
* [How to setup Perl ROUGE](http://kavita-ganesan.com/rouge-howto)
* [More details about plugins for Perl ROUGE](http://www.summarizerman.com/post/42675198985/figuring-out-rouge)

**Note:** As of 18th May 2017 the [website](http://berouge.com/) for the official Perl package appears to be down. Unfortunately you need to download a directory called `ROUGE-1.5.5` from there. As an alternative, it seems that you can get that directory from [here](https://github.com/andersjo/pyrouge) (however, the version of `pyrouge` in that repo appears to be outdated, so best to install `pyrouge` from the [official source](https://pypi.python.org/pypi/pyrouge)).

### Tensorboard
Run Tensorboard from the experiment directory (in the example above, `myexperiment`). You should be able to see data from the train and eval runs. If you select "embeddings", you should also see your word embeddings visualized.

### Help, I've got NaNs!
For reasons that are [difficult to diagnose](https://github.com/abisee/pointer-generator/issues/4), NaNs sometimes occur during training, making the loss=NaN and sometimes also corrupting the model checkpoint with NaN values, making it unusable. Here are some suggestions:

* If training stopped with the `Loss is not finite. Stopping.` exception, you can just try restarting. It may be that the checkpoint is not corrupted.
* You can check if your checkpoint is corrupted by using the `inspect_checkpoint.py` script. If it says that all values are finite, then your checkpoint is OK and you can try resuming training with it.
* The training job is set to keep 3 checkpoints at any one time (see the `max_to_keep` variable in `run_summarization.py`). If your newer checkpoint is corrupted, it may be that one of the older ones is not. You can switch to that checkpoint by editing the `checkpoint` file inside the `train` directory.
* Alternatively, you can restore a "best model" from the `eval` directory. See the note **Restoring snapshots** above.
* If you want to try to diagnose the cause of the NaNs, you can run with the `--debug=1` flag turned on. This will run [Tensorflow Debugger](https://www.tensorflow.org/versions/master/programmers_guide/debugger), which checks for NaNs and diagnoses their causes during training.
