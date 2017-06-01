from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cPickle
import numpy as np
import tensorflow as tf
import os
import time
import sys

from fwm import FastWeights
from snli_client import SNLIClient
import data_utils

import pdb


FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer("input_dim", 50, "Total (concatenated) allowed sentence length.")
tf.app.flags.DEFINE_integer("num_epochs", 200, "Max number of epochs to run.")
tf.app.flags.DEFINE_integer("batch_size", 256, "Batch size training and val.")
tf.app.flags.DEFINE_integer("steps_per_checkpoint", 200, "Number training steps per checkpoint.")
tf.app.flags.DEFINE_integer("validation_frequency", 1000, "Run validation after this many train steps.")
tf.app.flags.DEFINE_string("checkpoint_dir", "./checkpoints/", "Checkpoint directory.")
tf.app.flags.DEFINE_string("tb_dir", "./tb_logs", "Tensorboard log directory.")
tf.app.flags.DEFINE_string("data_dir", "./assets/", "Directory containing SNLI and embeddings (see SNLIClient).")
tf.app.flags.DEFINE_string("glove_fname", "precomputed_glove.weights", "Filename of your GloVe vocab embedding (see SNLICLient for script to generate).")
tf.app.flags.DEFINE_string("small_train_fname", "", "Optional mini/ custom test file for experimentation")
tf.app.flags.DEFINE_integer("embed_dim", 300, "Dimensions of emebeddings, i.e. GloVe 300d.")
tf.app.flags.DEFINE_integer("vocab_size", 42394, "Size of your vocab you prepared your embeddings with.")
tf.app.flags.DEFINE_integer("translate_layers", 2, "How many RNN layers pre concatenation.")
tf.app.flags.DEFINE_integer("translate_layer_units", 300, "Layer size for translate RNNs.")
tf.app.flags.DEFINE_integer("hidden_units", 256, "Number of hidden units in Fast Weights layer")
tf.app.flags.DEFINE_integer("s_loop", 1, "Number of iterative loops between hidden states.")
tf.app.flags.DEFINE_float("prelu_multiplier", 0.42, "Prelu multiplier internal to prelu implementation.")
tf.app.flags.DEFINE_float("fw_decay_l", 0.9, "Decay lambda used in Fast Weights implementation.")
tf.app.flags.DEFINE_float("fw_lr", 0.1, "Learning rate used in Fast Weights implementation.")
tf.app.flags.DEFINE_float("learning_rate", 0.0001, "Learning rate for optimizer.")
tf.app.flags.DEFINE_float("lr_decay_factor", 0.99, "Learning rate decay for optimizer.")
tf.app.flags.DEFINE_float("max_gradient_norm", 4.46, "Norm for clipping gradients.")

tf.app.flags.DEFINE_boolean("generate_glove_matrix", False, "Set True to generate GloVe SNLI vocab embedding matrix from common crawl.")
tf.app.flags.DEFINE_boolean("test", False, "Set to True for test mode.")

tf.logging.set_verbosity(tf.logging.INFO)

embedding_matrix = np.load(FLAGS.data_dir + FLAGS.glove_fname)

snli_client = SNLIClient(
    FLAGS.input_dim,
    FLAGS.data_dir,
    FLAGS.small_train_fname,
    FLAGS.glove_fname,
    FLAGS.generate_glove_matrix,
    FLAGS.test)

data_utils.prepare_datasets(snli_client)


def run_epoch(session, model, saver, num_steps, epoch_i, summ_writer):
    """ Run the model on the given data for an epoch

    Nans can occur easily and will stop training

    Fetches norm for inspection (but not printed)

    Experimentally batch norm functions better during validation the longer trianing has run
    """
    batch_losses = []; batch_accuracies = []
    current_step = 0
    step_time = 0.0

    for step_i in range(num_steps):

        start_time_step = time.time()

        batchX1, batchX2, batch_y = data_utils.get_batch(snli_client, FLAGS.batch_size)

        input_feed = {model.X1: batchX1, model.X2: batchX2, model.y: batch_y, model.phase: True}
        output_feed = [model.merged_summary, model.loss, model.accuracy, model.norm, model.sent1outa, model.sent2outb, model.update]

        summary, loss, accuracy, norm, output1, output2, _ = session.run(output_feed, input_feed)        

        if np.isnan(loss) or np.isnan(norm):
            pdb.set_trace()
            raise ValueError("Stopping - loss and norm values are nan.")

        summ_writer.add_summary(summary, epoch_i * num_steps + step_i)

        batch_losses.append(loss)
        batch_accuracies.append(accuracy)
        step_time += (time.time() - start_time_step)

        if current_step % FLAGS.steps_per_checkpoint == 0 and current_step != 0:
            print("stats for current step {}".format(current_step))
            print("TRAINING --> steps-checkpoint time: {:.4f}, loss: {:.4f}, accuracy: {:.4f}".format(step_time, loss, accuracy, norm))

            # Decrease learning rate if no improvement was seen over last 3 times.
            if len(batch_losses) > 2 and loss > max(batch_losses[-3:]):
                model.decay_lr(session, FLAGS.lr_decay_factor)

            if not os.path.exists(FLAGS.checkpoint_dir): os.mkdir(FLAGS.checkpoint_dir)
            saver.save(sess, FLAGS.checkpoint_dir, global_step=epoch_i*num_steps + step_i)

            if current_step % FLAGS.validation_frequency == 0 and current_step != 0:
                print("VALIDATION stats for current step {}".format(current_step))

                batchX1, batchX2, batch_y = data_utils.get_batch(snli_client, FLAGS.batch_size, mode="val")

                input_feed = {model.X1: batchX1, model.X2: batchX2, model.y: batch_y, model.phase: False}
                output_feed = [model.merged_summary, model.loss, model.accuracy, model.norm]

                summary, loss, accuracy, norm = session.run(output_feed, input_feed)

                if np.isnan(loss) or np.isnan(norm):
                    raise ValueError("Stopping - loss and norm values are nan.")

                val_losses.append(loss)
                val_accuracies.append(accuracy)
                print("VALIDATION --> loss: {:.6f}, acc: {:.6f}".format(val_losses[-1], val_accuracies[-1]))
            sys.stdout.flush()

        current_step += 1

    return batch_losses, batch_accuracies


def create_model(session):
    model = FastWeights(
        FLAGS.input_dim,
        FLAGS.fw_decay_l,
        FLAGS.fw_lr,
        FLAGS.vocab_size,
        FLAGS.embed_dim,
        FLAGS.batch_size,
        FLAGS.hidden_units,
        FLAGS.max_gradient_norm,
        FLAGS.prelu_multiplier,
        FLAGS.translate_layers,
        FLAGS.translate_layer_units,
        FLAGS.s_loop)
    # ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
    # if ckpt and tf.gfile.Exists(ckpt.model_checkpoint_path):
    #     print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
    #     saver = tf.train.Saver(tf.global_variables())
    #     saver.restore(session, ckpt.model_checkpoint_path)
    # else:
    #     print("Creating model with fresh parameters.")
    #     session.run(tf.global_variables_initializer())
    return model

def save_or_restore(sess):
    saver = tf.train.Saver(max_to_keep=5)
    if os.path.exists(FLAGS.checkpoint_dir) and os.listdir(FLAGS.checkpoint_dir) != []:
        print("\nrestoring model parameters\n")
        # if want to import graph
        # saver = tf.train.import_meta_graph('rntn_flat_tree_model-{:d}.meta'.format(step))
        saver.restore(sess, tf.train.latest_checkpoint(FLAGS.checkpoint_dir))
    else:
        print("\ncreating model with fresh parameters.\n")
    return saver


epoch_losses = []; val_losses = []
epoch_accuracies = []; val_accuracies = []

def train():

    with tf.Graph().as_default(), tf.Session() as session:

        model = create_model(session)

        session.run(tf.global_variables_initializer())

        model.assign_embeddings(session, embedding_matrix)

        model.assign_lr(session, FLAGS.learning_rate)

        saver = save_or_restore(session)

        summary_writer = tf.summary.FileWriter(FLAGS.tb_dir, graph=tf.get_default_graph())

        start_time = time.time()

        num_steps = np.int(np.float(data_utils.get_amount_data(snli_client))/np.float(FLAGS.batch_size))
        print("Num steps / epoch: {}".format(num_steps))

        for epoch_i in range(FLAGS.num_epochs):

            print("\n\n ---- EPOCH:", epoch_i+1)

            batch_losses, batch_accuracies = run_epoch(session, model, saver, num_steps, epoch_i, summary_writer)

            epoch_losses.append(np.mean(batch_losses))
            epoch_accuracies.append(np.mean(batch_accuracies))

            print("EPOCH {} --> time: {:.4f}, loss: {:.6f}, acc: {:.6f}".format(epoch_i,
                time.time() - start_time, epoch_losses[-1], epoch_accuracies[-1]))


def test():
    #TODO add mode for one-off testing logits -> labels, i.e. any sentence pair outputting 'contradict', 'entail', 'neutral'

    with tf.Graph().as_default(), tf.Session() as session:

        model = create_model(session)

        saver = save_or_restore(session)

        accuracies = []
        num_steps = np.int(np.float(data_utils.get_amount_data(snli_client, mode="test"))/np.float(FLAGS.batch_size))
        print("Num test steps: {}".format(num_steps))
        for step_i in range(num_steps):

            start_time_step = time.time()

            batchX1, batchX2, batch_y = data_utils.get_batch(snli_client, FLAGS.batch_size, mode="test")

            # model.logits for label
            accuracy = session.run(model.accuracy, feed_dict={model.X1: batchX1, model.X2: batchX2, model.y: batch_y, model.phase: False})

            accuracies.append(accuracy)
            step_time += (time.time() - start_time_step)

        print("Test accuracy: {:.6f} in time {:.4f} ".format(np.mean(accuracies)), step_time)



def main(_):
  if FLAGS.test:
    self_test()
  else:
    train()

if __name__ == "__main__":
  tf.app.run()
