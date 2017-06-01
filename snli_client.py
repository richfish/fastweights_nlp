from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import json
import os
import tempfile

from keras.preprocessing.text import Tokenizer
from keras.utils import np_utils
from keras.preprocessing.sequence import pad_sequences

_PAD = "PAD"
_GO = "GO"
_EOS = "EOS"
_UNK = "UNK"
_START_VOCAB = [_PAD, _GO, _EOS, _UNK]


class SNLIClient():

    def __init__(self, input_dim, data_dir="", small_train_fname="", embed_hidden_size=300,
                 generate_glove_matrix=False, test_only=False):
        """
        input_dim: dimension of eventual premise/ hypo. sentences concatenated (will divide by 2)
        data_dir: directeory where both your SNLI files and GloVe embeddings live
        small_train_fname: test with small/ tiny/ custom training size
        embed_hidden_size: dims of desired GloVe per word embeddings
        generate_glove_matrix: generate vocab-index matrix of GloVe embeddings
        test_only: only deal with test data
        """
        self.input_dim = int(input_dim/2)
        self.data_dir = data_dir
        self.train_txt_store = self.data_dir + "/snli_1.0_train.jsonl"
        self.use_small_file = bool(small_train_fname)
        self.train_txt_store_short = self.data_dir + "/" + small_train_fname
        self.dev_txt_store = self.data_dir + "/snli_1.0_dev.jsonl"
        self.test_txt_store = self.data_dir + "/snli_1.0_test.jsonl"
        self.embed_hidden_size = embed_hidden_size
        self.test_only = test_only
        self.snli_labels = {'contradiction': 0, 'neutral': 1, 'entailment': 2}
        self.processed = { "train": [], "val": [], "test": []}


    def build_and_save_glove_embeddings(self, glove_filename="precomputed_glove.weights"):
        # Using https://nlp.stanford.edu/projects/glove/ common crawl 840B tokens

        self.glove_store = self.data_dir + "/" + glove_filename
        vocab_index_map = self._vocab_index_map()
        if not os.path.exists(self.glove_store + '.npy'):
            embeddings_index = {}
            glove_path = os.path.join(self.data_dir, "glove.840B.300d.txt")
            f = open(glove_path)
            for line in f:
                values = line.split(' ')
                word = values[0]
                coefs = np.asarray(values[1:], dtype='float32')
                embeddings_index[word] = coefs
            f.close()

            vocab_size = len(vocab_index_map)
            embedding_matrix = np.zeros((vocab_size, self.embed_hidden_size))
            for i, word in vocab_index_map.items():
                embedding_vector = embeddings_index.get(word)
                if embedding_vector is None:
                    embedding_matrix[i] = embeddings_index['UNK']
                else:
                    embedding_matrix[i] = embedding_vector
            print("matrix done - size {}".format(len(embedding_matrix)))
            np.save(self.glove_store, embedding_matrix)


    def prepare_datasets(self, seq_len=None, use_short=False):
        if seq_len is None:
            seq_len = self.input_dim
        if self.use_small_file:
            use_short = True
        train, dev, test = self._prepare_datasets(seq_len, use_short)
        return train, dev, test

    def prepare_datasets_no_instr_chars(self, seq_len=None, use_short=False):
        if seq_len is None:
            seq_len = self.input_dim
        train, dev, test, _ = self._prepare_datasets_no_instr_chars(seq_len, use_short)
        return train, dev, test

    def prepare_only_val_test(self, seq_len=None, use_short=False):
        if seq_len is None:
            seq_len = self.input_dim
        dev, test = self._prepare_only_val_test(seq_len, use_short)
        return dev, test


    def write_tfrecords(self):
        if not os.path.exists(self.train_txt_store + 'train.tfrecords'):
            train, dev, test = self.train_dev_test()
            self._write_tfrecords(self.train_txt_store, 'train', train)
            self._write_tfrecords(self.dev_txt_store, 'dev', dev)
            self._write_tfrecords(self.test_txt_store, 'test', test)
        else:
            print("Already have tfrecords; remove to regenerate.")

    def _write_tfrecords(self, path, name, data):
        left, right, label = data[0], data[1], data[2]
        filename = os.path.join(self.data_dir, name + '.tfrecords')
        writer = tf.python_io.TFRecordWriter(filename)
        for i in range(len(left)):
            ex = self._make_tfrecord_example(left[i], right[i], label[i])
            writer.write(ex.SerializeToString())
        writer.close()

    def _make_tfrecord_example(self, sent1, sent2, labels):
        ex = tf.train.SequenceExample()
        seq_length = len(sent1)
        ex.context.feature["length"].int64_list.value.append(seq_length)
        # Feature lists for the two sequential features of our example
        s1_tokens = ex.feature_lists.feature_list["s1_tokens"]
        s2_tokens = ex.feature_lists.feature_list["s2_tokens"]
        fl_labels = ex.feature_lists.feature_list["labels"]
        for s1_token, s2_token, label in zip(sent1, sent2, labels):
            s1_tokens.feature.add().int64_list.value.append(np.int64(s1_token))
            s2_tokens.feature.add().int64_list.value.append(np.int64(s2_token))
            fl_labels.feature.add().int64_list.value.append(np.int64(label))
        return ex

    def _vocab_index_map(self):
        tokenizer = Tokenizer(lower=False, filters='')
        training = self._u_to_str(self._get_data(self.train_txt_store))
        tokenizer.fit_on_texts(training[0] + training[1])

        vocab_index_map = {}
        for i,word in enumerate(_START_VOCAB):
            vocab_index_map[i] = word
        for word, word_i in tokenizer.word_index.items():
            vocab_index_map[word_i+3] = word
        return vocab_index_map

    def _fit_tokenizer(self):
        _, _, _, tokenizer = self._train_dev_test_tokenizer()
        return tokenizer


    def _extract_tokens_from_binary_parse(self, parse):
        return parse.replace('(', ' ').replace(')', ' ').replace('-LRB-', '(').replace('-RRB-', ')').split()

    def _u_to_str(self, multi_tuple):
        tlist = list(multi_tuple)
        for i in range(len(multi_tuple)-1):
            tlist[i] = [str(x) for x in multi_tuple[i]]
        return tuple(tlist)


    def _prepare_datasets(self, seq_len, use_short=False):
        """
        For all datasets returns arryays of vocab-indexed sents with padding;
        Embedding with GloVe happens during assignment in training

        seq_len: length you want premise/ hyp. sentences to be (pre concatenation)
        use_short: test a tiny version of the model

        Returns:
        Triples for each dataset (premise, hypothesis, label)
        """

        file_source = self.train_txt_store_short if use_short else self.train_txt_store
        training = self._u_to_str(self._get_data(file_source))
        dev = self._u_to_str(self._get_data(self.dev_txt_store))
        test = self._u_to_str(self._get_data(self.test_txt_store))

        tokenizer = Tokenizer(lower=False, filters='')
        tokenizer.fit_on_texts(training[0] + training[1])

        to_seq = lambda X: pad_sequences(tokenizer.texts_to_sequences(X), maxlen=seq_len)
        prepare_data = lambda data: (to_seq(data[0]), to_seq(data[1]), data[2])

        training = prepare_data(training)
        dev = prepare_data(dev)
        test = prepare_data(test)

        # shift for instruction chars
        sents1 = training[0]
        sents2 = training[1]
        sents1_final = []
        sents2_final = []
        for s1, s2 in zip(sents1, sents2):
            shifted1 = [x+3 if x != 0 else x for x in s1]
            shifted2 = [x+3 if x != 0 else x for x in s2]
            sents1_final.append(shifted1)
            sents2_final.append(shifted2)
            # NOTE if wanted to tweak to embed here:
            # sents1_final.append([embedding_matrix[x] for x in shifted1]);
        training = [sents1_final, sents2_final, training[2]]

        sents1_dev = dev[0]
        sents2_dev = dev[1]
        sents1_dev_final = []
        sents2_dev_final = []
        for s1, s2 in zip(sents1_dev, sents2_dev):
            shifted1 = [x+3 if x != 0 else x for x in s1]
            shifted2 = [x+3 if x != 0 else x for x in s2]
            sents1_dev_final.append(shifted1)
            sents2_dev_final.append(shifted2)
        dev = [sents1_dev_final, sents2_dev_final, dev[2]]

        sents1_test = test[0]
        sents2_test = test[1]
        sents1_test_final = []
        sents2_test_final = []
        for s1, s2 in zip(sents1_test, sents2_test):
            shifted1 = [x+3 if x != 0 else x for x in s1]
            shifted2 = [x+3 if x != 0 else x for x in s2]
            sents1_test_final.append(shifted1)
            sents2_test_final.append(shifted2)
        test = [sents1_test_final, sents2_test_final, test[2]]

        return training, dev, test


    def _prepare_only_val_test(self, seq_len, use_short=False):
        """
        If want to skip time it takes to load train
        """
        file_source = self.train_txt_store_short if use_short else self.train_txt_store
        training = self._u_to_str(self._get_data(file_source))
        dev = self._u_to_str(self._get_data(self.dev_txt_store))
        test = self._u_to_str(self._get_data(self.test_txt_store))

        tokenizer = Tokenizer(lower=False, filters='')
        tokenizer.fit_on_texts(training[0] + training[1])

        to_seq = lambda X: pad_sequences(tokenizer.texts_to_sequences(X), maxlen=seq_len)
        prepare_data = lambda data: (to_seq(data[0]), to_seq(data[1]), data[2])

        dev = prepare_data(dev)
        test = prepare_data(test)

        sents1_dev = dev[0]
        sents2_dev = dev[1]
        sents1_dev_final = []
        sents2_dev_final = []
        for s1, s2 in zip(sents1_dev, sents2_dev):
            shifted1 = [x+3 if x != 0 else x for x in s1]
            shifted2 = [x+3 if x != 0 else x for x in s2]
            sents1_dev_final.append(shifted1)
            sents2_dev_final.append(shifted2)
        dev = [sents1_dev_final, sents2_dev_final, dev[2]]

        sents1_test = test[0]
        sents2_test = test[1]
        sents1_test_final = []
        sents2_test_final = []
        for s1, s2 in zip(sents1_test, sents2_test):
            shifted1 = [x+3 if x != 0 else x for x in s1]
            shifted2 = [x+3 if x != 0 else x for x in s2]
            sents1_test_final.append(shifted1)
            sents2_test_final.append(shifted2)
        test = [sents1_test_final, sents2_test_final, test[2]]

        return dev, test


    def _prepare_datasets_no_instr_chars(self, seq_len, user_short=False):
        file_source = self.train_txt_store_short if use_short else self.train_txt_store
        training = self._u_to_str(self._get_data(file_source))
        dev = self._u_to_str(self._get_data(self.dev_txt_store))
        test = self._u_to_str(self._get_data(self.test_txt_store))

        tokenizer = Tokenizer(lower=False, filters='')
        tokenizer.fit_on_texts(training[0] + training[1])

        to_seq = lambda X: pad_sequences(tokenizer.texts_to_sequences(X), maxlen=seq_len)
        prepare_data = lambda data: (to_seq(data[0]), to_seq(data[1]), data[2])

        training = prepare_data(training)
        dev = prepare_data(dev)
        test = prepare_data(test)

        return training, dev, test, tokenizer


    def _yield_examples(self, fn, skip_no_majority=True, limit=None):
        for i, line in enumerate(open(fn)):
            if limit and i > limit:
              break
            data = json.loads(line)
            label = data['gold_label']
            s1 = ' '.join(self._extract_tokens_from_binary_parse(data['sentence1_binary_parse']))
            s2 = ' '.join(self._extract_tokens_from_binary_parse(data['sentence2_binary_parse']))
            if skip_no_majority and label == '-':
                continue
            yield (label, s1, s2)


    def _get_data(self, fn, limit=None):
        raw_data = list(self._yield_examples(fn=fn))
        left = [s1 for _, s1, s2 in raw_data]
        right = [s2 for _, s1, s2 in raw_data]
        # print(max(len(x.split()) for x in left))
        # print(max(len(x.split()) for x in right))
        Y = np.array([self.snli_labels[l] for l, s1, s2 in raw_data])
        Y = np_utils.to_categorical(Y, len(self.snli_labels))
        return left, right, Y
