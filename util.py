# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
# Modifications Copyright 2017 Abigail See
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

"""This file contains some utility functions"""
import re

import tensorflow as tf
import time
import os
from os.path import basename

import gensim
import numpy as np
import boto3
from boto3.dynamodb.conditions import Key

from IPython import embed
from tqdm import tqdm

import asyncio
from mmcore.clients.word2vec.client import W2VLambdaClient

import pickle

FLAGS = tf.app.flags.FLAGS

# For chunking the query list
def chunks_list(array, chunk_size):
    chunked_list = []
    for i in range(0, len(array), chunk_size):
        chunked_list.append(array[i:i + chunk_size])
    return chunked_list

def get_config():
    """Returns config for tf.session"""
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth=True
    return config

def load_ckpt(saver, sess, ckpt_dir="train"):
    """Load checkpoint from the ckpt_dir (if unspecified, this is train dir) and restore it to saver and sess, waiting 10 secs in the case of failure. Also returns checkpoint name."""
    while True:
        try:
            latest_filename = "checkpoint_best" if ckpt_dir=="eval" else None
            ckpt_dir = os.path.join(FLAGS.log_root, ckpt_dir)
            ckpt_state = tf.train.get_checkpoint_state(ckpt_dir, latest_filename=latest_filename)
            tf.logging.info('Loading checkpoint %s', ckpt_state.model_checkpoint_path)
            saver.restore(sess, ckpt_state.model_checkpoint_path)
            return ckpt_state.model_checkpoint_path
        except:
            tf.logging.info("Failed to load checkpoint from %s. Sleeping for %i secs...", ckpt_dir, 10)
            time.sleep(10)

def make_cc_embedding_old(id2word):

    print("Retrieving common crawl vectors")
    vocab_size = len(id2word)
    emb_dim = 300

    dynamodb = boto3.resource('dynamodb', aws_access_key_id=os.environ['AWS_ACCESS_KEY'], aws_secret_access_key=os.environ['AWS_SECRET_KEY'], region_name=os.environ['AWS_REGION'])
    table = dynamodb.Table('W2V_CC_1.7M')
    embeddings = np.random.rand(vocab_size, emb_dim)
    for ind, word in tqdm(id2word.items()):
        #print(word)
        #resp = table.get_item(Key={'word': word})

        item = table.query(KeyConditionExpression=Key('word').eq(word))
        if len(item['Items']) == 0:
            continue

        item = item['Items'][0]
        vector = np.array(list(map(float, item['vector'].split(" "))))
        # embed()
        # exit()
        embeddings[ind, :] = vector

    return embeddings




def make_cc_embedding(vocab):

    vocab_size = vocab._count
    emb_dim = 300
    vocabulary = list(vocab._word_to_id.keys())
    embeddings = np.random.rand(vocab_size, emb_dim)

    #embed()
    if os.path.exists(FLAGS.emb_path):
        print("Retrieving common crawl vectors from pickle.......", end="", flush=True)
        with open(FLAGS.emb_path, 'rb') as r:
            emb_dict = pickle.load(r)

        for word, vec in emb_dict.items():
            ind = vocab._word_to_id[word]
            embeddings[ind, :] = vec
        print("done")
    else:
        print("Retrieving common crawl vectors from w2v_client.......", end="", flush=True)
        loop = asyncio.get_event_loop()
        w2v_client = W2VLambdaClient(loop)

        # vocab = list of words
        chunked_vocab = chunks_list(vocabulary, 400)

        requests = []
        counter = 0
        vec_results = {}
        for vocab_chunk in chunked_vocab:
            counter += 1
            requests.append(w2v_client.vectors(keywords=vocab_chunk, vectors='vecs_unnormalized', holistic_vectors=True))

            if counter % 10 == 0:

                results = loop.run_until_complete(
                    asyncio.gather(*requests, return_exceptions=True))

                requests = []

                for res in results:
                    for word, vec in res.items():
                        vec_results[word] = vec
                        ind = vocab._word_to_id[word]
                        embeddings[ind, :] = vec

        if len(requests) > 0:
            results = loop.run_until_complete(
                asyncio.gather(*requests, return_exceptions=True))

            requests = []

            for res in results:
                for word, vec in res.items():
                    vec_results[word] = vec
                    ind = vocab._word_to_id[word]
                    embeddings[ind, :] = vec
        print("done")
        print("Writing common crawl vectors to the file {} .......".format(FLAGS.emb_path), end="", flush=True)
        try:
            with open(FLAGS.emb_path, 'wb') as w:
                pickle.dump(vec_results, w)
        except:
            if os.path.exists(FLAGS.emb_path):
                os.remove(FLAGS.emb_path)

        print("done")

    return embeddings

    # loop = asyncio.get_event_loop()


    # # vocab = list of words
    # chunked_vocab = chunks_list(vocab, 400)

    # requests = []
    # counter = 0
    # vec_results = {}
    # for vocab_chunk in vocab:
    #     counter += 1
    #     requests.append(w2v_client.get_vectors(word_chunk, vectors='vecs_unnormalized', holistic_vectors=True))

    #     if counter % 10 == 0:


    #         results = loop.run_until_complete(
    #             asyncio.gather(requests, return_exceptions=True))

    #         requests = []

    #         for res in results:
    #             for word, vec in res.items():
    #                 vec_results[word] = vec
