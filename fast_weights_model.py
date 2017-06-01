from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

class FastWeights(object):

    def __init__(self, input_dim, fw_decay_l, fw_lr, vocab_size, embed_dim,
                batch_size, hidden_units, max_gradient_norm, prelu_multiplier,
                translate_layers, translate_layer_units, s_loop, tb_logging=True):

        self.component_input_dim = input_dim/2
        self.num_labels = 3 # entailment, contradiction, neutral
        self.input_dim = input_dim
        self.fw_defcay_l = fw_decay_l
        self.fw_lr = fw_lr
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.batch_size = batch_size
        self.hidden_units = hidden_units
        self.max_gradient_norm = max_gradient_norm
        self.prelu_multiplier = prelu_multiplier
        self.rnn_translate_layers = translate_layers
        self.translate_layer_units = translate_layer_units
        self.s_loop = s_loop

        self.build_core_objs()
        self.run_rnn_translate()
        self.run_fast_weights_layer()
        self.run_computations()
        self.log_tensorboard()

    def run_computations(self):
        with tf.name_scope('Logits'):
            self.logits = tf.matmul(self.h, self.W_softmax) + self.b_softmax

        with tf.name_scope('Loss'):
            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.y))

        self.lr = tf.Variable(0.0, trainable=False)

        self.global_step = tf.Variable(0, trainable=False)

        # return and set global norm for stability checks
        with tf.name_scope('Grads_norms_optimize'):
            opt = self._get_optimizer()
            params = tf.trainable_variables()
            gradients = tf.gradients(self.loss, params)
            self.grads, self.norm = tf.clip_by_global_norm(gradients, self.max_gradient_norm)
            self.update = opt.apply_gradients(zip(self.grads, params))

        with tf.name_scope('Accuracy'):
            self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.logits, 1),
                tf.argmax(self.y, 1)), tf.float32))


    def build_core_objs(self):

        self.X1 = tf.placeholder(tf.int32, [None, self.component_input_dim])
        self.X2 = tf.placeholder(tf.int32, [None, self.component_input_dim])
        self.y = tf.placeholder(tf.float32, shape=[None, self.num_labels], name='targets_y')
        #self.l = tf.placeholder(tf.float32, [], name="fw_learning_rate")
        #self.e = tf.placeholder(tf.float32, [], name="fw_decay_rate")
        self.phase = tf.placeholder(tf.bool, name='phase') #batch norm

        with tf.name_scope("embedding_sent1"):
            self.W_embed_sent1 = tf.Variable(tf.random_uniform([self.vocab_size, self.embed_dim], -1.0, 1.0), name="W")
            self.embedded_chars_sent1 = tf.nn.embedding_lookup(self.W_embed_sent1, self.X1)
        with tf.name_scope("embedding_sent2"):
            self.W_embed_sent2 = tf.Variable(tf.random_uniform([self.vocab_size, self.embed_dim], -1.0, 1.0), name="W")
            self.embedded_chars_sent2 = tf.nn.embedding_lookup(self.W_embed_sent2, self.X2)

        self.Xret1 = self.embedded_chars_sent1
        self.Xret2 = self.embedded_chars_sent2

        with tf.variable_scope('rnn-translate-1'):
            self.softmax_w1 = tf.get_variable("softmax_w1", [self.embed_dim, self.translate_layer_units])
            self.softmax_b1 = tf.get_variable("softmax_b1", [self.translate_layer_units])
        with tf.variable_scope('rnn-translate-2'):
            self.softmax_w2 = tf.get_variable("softmax_w2", [self.embed_dim, self.translate_layer_units])
            self.softmax_b2 = tf.get_variable("softmax_b2", [self.translate_layer_units])

        with tf.variable_scope("fast_weights"):
            self.W_x = tf.Variable(tf.random_uniform([self.translate_layer_units, self.hidden_units], -np.sqrt(2.0/self.translate_layer_units), np.sqrt(2.0/self.translate_layer_units)), dtype=tf.float32)
            self.b_x = tf.Variable(tf.zeros([self.hidden_units]), dtype=tf.float32)
            self.W_h = tf.Variable(initial_value=0.05 * np.identity(self.hidden_units), dtype=tf.float32)

            self.W_softmax = tf.Variable(tf.random_uniform([self.hidden_units, self.num_labels], -np.sqrt(2.0 / self.hidden_units), np.sqrt(2.0 / self.hidden_units)), dtype=tf.float32)
            self.b_softmax = tf.Variable(tf.zeros([self.num_labels]), dtype=tf.float32)

            # scale and shift for layernorm
            self.gain = tf.Variable(tf.ones([self.hidden_units]), dtype=tf.float32)
            self.bias = tf.Variable(tf.zeros([self.hidden_units]), dtype=tf.float32)

        # fast weights matrix
        self.A = tf.zeros([self.batch_size, self.hidden_units, self.hidden_units], dtype=tf.float32)

        self.h = tf.zeros([self.batch_size, self.hidden_units], dtype=tf.float32)



    def run_rnn_translate(self):
        """ Run in parallel for both premise and hypothesis sentences.
        """

        _X1 = tf.transpose(self.Xret1, [1, 0, 2])
        _X1 = tf.reshape(_X1, [-1, self.embed_dim])
        _X1 = self._parametric_relu(tf.matmul(_X1, self.softmax_w1) + self.softmax_b1, layeri=1)
        _X1 = tf.split(axis=0, num_or_size_splits=int(self.component_input_dim), value=_X1)

        _X2 = tf.transpose(self.Xret2, [1, 0, 2])
        _X2 = tf.reshape(_X2, [-1, self.embed_dim])
        _X2 = self._parametric_relu(tf.matmul(_X2, self.softmax_w2) + self.softmax_b2, layeri=2)
        _X2 = tf.split(axis=0, num_or_size_splits=int(self.component_input_dim), value=_X2)

        self.rnn_cell1 = tf.contrib.rnn.BasicRNNCell(self.translate_layer_units)
        self.multi_rnn1 = tf.contrib.rnn.MultiRNNCell([self.rnn_cell1] * self.rnn_translate_layers, state_is_tuple=True)
        self.rnn_cell2 = tf.contrib.rnn.BasicRNNCell(self.translate_layer_units)
        self.multi_rnn2 = tf.contrib.rnn.MultiRNNCell([self.rnn_cell2] * self.rnn_translate_layers, state_is_tuple=True)

        with tf.variable_scope('premise-rnn'):
            self.outputs1, self.states1 = tf.contrib.rnn.static_rnn(self.multi_rnn1, _X1, dtype=tf.float32)
        with tf.variable_scope('hypothesis-rnn'):
            self.outputs2, self.states2 = tf.contrib.rnn.static_rnn(self.multi_rnn2, _X2, dtype=tf.float32)

        self.sent1outa = tf.transpose(self.outputs1, [1,0,2])
        self.sent2outb = tf.transpose(self.outputs2, [1,0,2])
        sent1out = self.sent1outa
        sent2out = self.sent2outb

        with tf.variable_scope("bn_sent1"):
            sent1out = tf.contrib.layers.batch_norm(sent1out, center=True, scale=True, is_training=self.phase)
        with tf.variable_scope("bn_sent2"):
            sent2out = tf.contrib.layers.batch_norm(sent2out, center=True, scale=True, is_training=self.phase)

        self.X = tf.concat(axis=1,values=[sent1out, sent2out])


    def run_fast_weights_layer(self):
        # Implementaiton largely borrowed from https://github.com/GokuMohandas/fast-weights/blob/master/fw/model.py
        for t in range(0, self.input_dim):

            self.h = self._parametric_relu((tf.matmul(self.X[:, t, :], self.W_x)+self.b_x) + (tf.matmul(self.h, self.W_h)), layeri=3)

            self.h_s = tf.reshape(self.h, [self.batch_size, 1, self.hidden_units])

            # FW matrix for this timestep
            self.A = tf.add(tf.scalar_mul(self.fw_defcay_l, self.A), tf.scalar_mul(self.fw_lr, tf.matmul(tf.transpose( self.h_s, [0, 2, 1]), self.h_s)))

            # FW matrix iterative iteraction loop w/ layernorm
            for _ in range(self.s_loop):
                self.h_s = tf.reshape(
                    tf.matmul(self.X[:, t, :], self.W_x)+self.b_x,
                    tf.shape(self.h_s)) + tf.reshape(
                    tf.matmul(self.h, self.W_h), tf.shape(self.h_s)) + \
                    tf.matmul(self.h_s, self.A)

                m,v = tf.nn.moments(self.h_s, axes=[0], keep_dims=True)
                normalized_input = (self.h_s - m) / tf.sqrt(v + 1e-5)
                self.h_s = normalized_input * self.gain + self.bias
                # alternative layernorm
                # self.mu = tf.reduce_mean(self.h_s, reduction_indices=0) # each sample
                # self.sigma = tf.sqrt(tf.reduce_mean(tf.square(self.h_s - self.mu), reduction_indices=0))
                # self.h_s = tf.div(tf.mul(self.gain, (self.h_s - self.mu)), self.sigma) + self.bias

                self.h_s = self._parametric_relu(self.h_s, layeri=4)
            self.h = tf.reshape(self.h_s, [self.batch_size, self.hidden_units])


    def log_tensorboard(self):
        """
        Expand as desire
        """
        tf.summary.scalar("loss", self.loss)
        tf.summary.scalar("accuracy", self.accuracy)
        self.merged_summary = tf.summary.merge_all()


    def assign_lr(self, session, lr_value):
        session.run(tf.assign(self.lr, lr_value))

    def decay_lr(self, session, decay_rate):
        session.run(tf.assign(self.lr, self.lr * decay_rate))

    def assign_embeddings(self, session, embedding_matrix):
        session.run(self.W_embed_sent1.assign(embedding_matrix))
        session.run(self.W_embed_sent2.assign(embedding_matrix))

    def _get_optimizer(self):
        return tf.train.AdamOptimizer(learning_rate=self.lr, epsilon=1e-4)

    def _parametric_relu(self, _x, layeri):
        try:
            with tf.variable_scope("alpha{}".format(layeri), reuse=True):
                alphas = tf.get_variable("alpha", _x.get_shape()[-1], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
        except Exception as e:
            with tf.variable_scope("alpha{}".format(layeri)):
                alphas = tf.get_variable("alpha", _x.get_shape()[-1], initializer=tf.constant_initializer(0.0), dtype=tf.float32)

        pos = tf.nn.relu(_x)
        neg = alphas * (_x - abs(_x)) * self.prelu_multiplier
        return pos + neg
