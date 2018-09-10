import os


import sys
import hashlib
import struct
import subprocess
import collections

import argparse
import tensorflow as tf
from tensorflow.core.example import example_pb2

from tqdm import tqdm

SENTENCE_START = '<s>'
SENTENCE_END = '</s>'

VOCAB_SIZE = 200000
CHUNK_SIZE = 1000 # num examples per chunk, for the chunked data

dm_single_close_quote = u'\u2019' # unicode
dm_double_close_quote = u'\u201d'
END_TOKENS = ['.', '!', '?', '...', "'", "`", '"', dm_single_close_quote, dm_double_close_quote, ")"] # acceptable ways to end a sentence

TRAIN_SPLIT = 0.80
VAL_SPLIT = 0.10
TEST_SPLIT = 0.10

PUNCS = '.,?!:;'


def chunk_file(set_name, chunks_dir, data_dir):
    in_file = os.path.join(data_dir, set_name+'.bin')
    reader = open(in_file, "rb")
    chunk = 0
    finished = False
    while not finished:
        chunk_fname = os.path.join(chunks_dir, '%s_%03d.bin' % (set_name, chunk)) # new chunk
        with open(chunk_fname, 'wb') as writer:
            for _ in range(CHUNK_SIZE):
                len_bytes = reader.read(8)
                if not len_bytes:
                    finished = True
                    break
                str_len = struct.unpack('q', len_bytes)[0]
                example_str = struct.unpack('%ds' % str_len, reader.read(str_len))[0]
                writer.write(struct.pack('q', str_len))
                writer.write(struct.pack('%ds' % str_len, example_str))
            chunk += 1


def chunk_all(chunks_dir, data_dir):
    # Make a dir to hold the chunks
    if not os.path.isdir(chunks_dir):
        os.mkdir(chunks_dir)
    # Chunk the data
    for set_name in ['train', 'val', 'test']:
        print("Splitting {} data into chunks...".format(set_name))
        chunk_file(set_name, chunks_dir, data_dir)
    print("Saved chunked data in %s" % chunks_dir)


def split_data(data):

    total_ex = len(data)
    train_ex = int(total_ex*TRAIN_SPLIT)
    val_ex = int(total_ex*VAL_SPLIT)

    train = data[:train_ex]
    val = data[train_ex:train_ex+val_ex]
    test = data[train_ex+val_ex:]

    return {"train": train, "val": val, "test": test}

def clean_example(example):
    for punc in PUNCS:
      splits = example.split(punc)
      cleaned = [' '+e+' ' if len(e) > 1 and e[0]!=' ' and e[-1]!=' ' else e+' ' if len(e) > 0 and e[-1]!=' ' else ' '+e if len(e)>0 and e[0]!=' ' else e for e in splits]
      example = punc.join(cleaned)
    return example

def main(args):

    if args.no_vocab:
        makevocab = False
    else:
        makevocab = True

    data_dir = args.data_dir

    files = os.listdir(data_dir)
    files = [e for e in files if e.split('.')[-1]=='tsv']


    for file in files:
        if makevocab:
            vocab_counter = collections.Counter()
        examples = []
        with open(os.path.join(data_dir, file)) as r:
            for entry in r:
                examples.append(entry)

        splits_dic = split_data(examples)

        if args.out_dir is None:
            main_out_dir = os.path.join(args.data_dir)
        else:
            main_out_dir = args.out_dir

        curr_out_dir = os.path.join(main_out_dir, file.split('.')[0], "finished_files")

        if not os.path.exists(curr_out_dir):
            os.makedirs(curr_out_dir)

        for k, v in splits_dic.items():
            
            num_stories = len(v)
            out_file = os.path.join(curr_out_dir, k+'.bin')

            with open(out_file, 'wb') as writer:
                for idx,s in enumerate(v):
                    if idx % 1000 == 0:
                        print("Writing paraphrase %i of %i; %.2f percent done" % (idx, num_stories, float(idx)*100.0/float(num_stories)))

                    ex_splits = s.split('\t')
                    tmp_splits = []
                    for ex in ex_splits:
                        tmp_splits.append(clean_example(ex)) 
                    article, abstract = tmp_splits[0], tmp_splits[1]
                    # Write to tf.Example
                    tf_example = example_pb2.Example()
                    tf_example.features.feature['article'].bytes_list.value.extend([article.encode()])
                    tf_example.features.feature['abstract'].bytes_list.value.extend([abstract.encode()])
                    tf_example_str = tf_example.SerializeToString()
                    str_len = len(tf_example_str)
                    writer.write(struct.pack('q', str_len))
                    writer.write(struct.pack('%ds' % str_len, tf_example_str))

                    # Write the vocab to file, if applicable
                    if makevocab:
                        art_tokens = article.split(' ')
                        abs_tokens = abstract.split(' ')
                        abs_tokens = [t for t in abs_tokens if t not in [SENTENCE_START, SENTENCE_END]] # remove these tags from vocab
                        tokens = art_tokens + abs_tokens
                        tokens = [t.strip() for t in tokens] # strip
                        tokens = [t for t in tokens if t!=""] # remove empty
                        vocab_counter.update(tokens)

            print("Finished writing file %s\n" % out_file)

            # write vocab to file
            if makevocab:
                print("Writing vocab file...")
                with open(os.path.join(curr_out_dir, "vocab"), 'w') as writer:
                  for word, count in vocab_counter.most_common(VOCAB_SIZE):
                    writer.write(word + ' ' + str(count) + '\n')
                print("Finished writing vocab file")

        c_dir = os.path.join(curr_out_dir, "chunked")
        chunk_all(c_dir, curr_out_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', help='directory of paraphrase ensemble data')
    parser.add_argument('--out_dir', help='output directory of data splits for pointer generator', default=None)
    parser.add_argument('--no_vocab', help='whether or not to create vocab', action="store_true")
    args = parser.parse_args()
    main(args)