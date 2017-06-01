#  Largely borrowed from Stephen Merity's Keras LSTM https://github.com/Smerity/keras_snli

from __future__ import print_function
from functools import reduce
import os
import re
import tempfile
import numpy as np
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers.normalization import BatchNormalization
from keras.layers.embeddings import Embedding
from keras.models import Model
from keras.layers import merge, recurrent, Dense, Input, Dropout, TimeDistributed
from keras.regularizers import l2

from snli_client import SNLIClient
import data_utils

EMBED_HIDDEN_SIZE = 300
MAX_LEN = 25
SENT_HIDDEN_SIZE = 300
BATCH_SIZE = 128
PATIENCE = 4
MAX_EPOCHS = 42
DP = 0.2
L2 = 4e-6
TRAIN_EMBED = False
LAYERS = 1
LABELS = {'contradiction': 0, 'neutral': 1, 'entailment': 2}
VOCAB = 42391 # Vocab size using for SNLI corpus, switch out if you want smaller
VOCAB_WITH_ICHARS = VOCAB+3 # Vocab with instruction characters
GLOVE_STORE = "path/to/glovestore.weights"
TRAIN_DIR = ""
DATA_DIR = ""
RNN = recurrent.LSTM

snli_client = SNLIClient(MAX_LEN, DATA_DIR)

data_utils.prepare_datasets(snli_client)

training = snli_client.processed['train']
validation = snli_client.processed['val']
test = snli_client.processed['test']

embedding_matrix = np.load(GLOVE_STORE + '.npy')
embed = Embedding(VOCAB_WITH_ICHARS, EMBED_HIDDEN_SIZE, weights=[embedding_matrix], input_length=MAX_LEN, trainable=TRAIN_EMBED)

rnn_kwargs = dict(output_dim=SENT_HIDDEN_SIZE, dropout_W=DP, dropout_U=DP)
translate = TimeDistributed(Dense(SENT_HIDDEN_SIZE, activation='relu'))

premise = Input(shape=(MAX_LEN,), dtype='int32')
hypothesis = Input(shape=(MAX_LEN,), dtype='int32')

prem = embed(premise)
hypo = embed(hypothesis)

prem = translate(prem)
hypo = translate(hypo)

if LAYERS > 1:
    for _ in range(LAYERS - 1):
        rnn = RNN(return_sequences=True, **rnn_kwargs)
        prem = BatchNormalization()(rnn(prem))
        hypo = BatchNormalization()(rnn(hypo))

rnn = RNN(return_sequences=False, **rnn_kwargs)
prem = rnn(prem)
hypo = rnn(hypo)
prem = BatchNormalization()(prem)
hypo = BatchNormalization()(hypo)

joint = merge([prem, hypo], mode='concat')
joint = Dropout(DP)(joint)
for i in range(3):
    joint = Dense(2 * SENT_HIDDEN_SIZE, activation='relu', W_regularizer=l2(L2) if L2 else None)(joint)
    joint = Dropout(DP)(joint)
    joint = BatchNormalization()(joint)

pred = Dense(output_dim=len(LABELS), activation='softmax')(joint)

model = Model(input=[premise, hypothesis], output=pred)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy']) #or rmsprop

_, tmpfn = tempfile.mkstemp()
callbacks = [EarlyStopping(patience=PATIENCE), ModelCheckpoint(tmpfn, save_best_only=True, save_weights_only=True)]

model.fit([training[0], training[1]], training[2], batch_size=BATCH_SIZE, nb_epoch=MAX_EPOCHS, validation_data=([validation[0], validation[1]], validation[2]), callbacks=callbacks)

# To test restore the best model and uncomment test evaluation line
# model.load_weights(path/to/model)
# loss, acc = model.evaluate([test[0], test[1]], test[2], batch_size=BATCH_SIZE)

loss, acc = model.evaluate([training[0], training[1]], training[2], batch_size=BATCH_SIZE)
print('Test loss / test accuracy = {:.4f} / {:.4f}'.format(loss, acc))
