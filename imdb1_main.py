# -*- coding: utf-8 -*-
import sys
import os
import logging
import argparse
import time
import numpy as np
time_format="%Y%m%d_%H%M%S"

# fix random seed for reproducibility
# Set only during development
seed = 7
np.random.seed(seed)

# Dependencias de módulos creados por mi para otros proyectos:
sys.path.insert(0, os.path.abspath("models"))
from yoon_model import TextCNN
from keras.datasets import imdb
from keras.callbacks import ModelCheckpoint, Callback
from keras.preprocessing import sequence

COLLECTION_PATH="imdb.npz"
VOCABULARY_SIZE = 10000
SENTENCE_LENGTH = 1600
EMBEDDING_LENGTH = 128
FILTERS = 128
FILTER_SIZES = [3, 4, 5]
EPOCHS = 2
BATCH_SIZE = 16
DROPOUT = 0.5
LOSS_FUNCTION='binary_crossentropy'
OPTIMIZER='adam'


def main(args):
	logging.info("Starting script")
	logging.info(
		"Parameters: [COLLECTION_PATH=%s,VOCABULARY_SIZE=%d, SENTENCE_LENGTH=%d, EMBEDDING_LENGTH=%d,FILTERS=%d,FILTER_SIZES=%s,EPOCHS=%d,BATCH_SIZE=%d,DROPOUT=%.2f,LOSS=%s,OPTIMIZER=%s]"
		% (COLLECTION_PATH,
		VOCABULARY_SIZE,
		SENTENCE_LENGTH,
		EMBEDDING_LENGTH,
		FILTERS,
		str(FILTER_SIZES),
		EPOCHS,
		BATCH_SIZE,
		DROPOUT,
		LOSS_FUNCTION,
		OPTIMIZER)
	)
	logging.info("Generating network")
	textCnn = TextCNN(SENTENCE_LENGTH,
		EMBEDDING_LENGTH,
		VOCABULARY_SIZE,
		FILTERS,
		FILTER_SIZES,
		DROPOUT)
	model = textCnn.model
	model.compile(loss=LOSS_FUNCTION, optimizer=OPTIMIZER, metrics=['accuracy'])
	#textCnn.plot_model('plots/yoon_architecture_binary_output')
	logging.info("Loading collection")
	(x_train, y_train), (x_test, y_test) = imdb.load_data(
		path=COLLECTION_PATH,
		num_words=VOCABULARY_SIZE,
		skip_top=0,
		maxlen=None,
		seed=113,
		start_char=1,
		oov_char=2,
		index_from=3
	)
	logging.info("Shapes: x_train=%s, y_train=%s, x_test=%s, y_test=%s" % (
		str(x_train.shape),
		str(y_train.shape),
		str(x_test.shape),
		str(y_test.shape),
		))

	logging.info("Encoding training and test sets")
	x_train = sequence.pad_sequences(x_train, maxlen=SENTENCE_LENGTH)
	x_test = sequence.pad_sequences(x_test, maxlen=SENTENCE_LENGTH)

	# checkpoint
	filepath = "keras_models/" + time.strftime(time_format) + "_" + "{epoch:02d}_{val_acc:.4f}.hdf5"
	checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
	# logs during training
	callback1 = MyCallback()
	callbacks_list = [checkpoint, callback1]

	model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=callbacks_list, verbose=0)
	model_filepath = "keras_models/" + time.strftime(time_format) + "_" + "%depochs_%dbatchsize_%dembeddings_%dfilters_" % (EPOCHS, BATCH_SIZE, EMBEDDING_LENGTH, FILTERS) + "_".join(str(ks) for ks in FILTER_SIZES) + "filtersize" + ".h5"
	model.save(model_filepath)



def loadArgParser():
	"""
	To handle flags used to run this script on console
	"""
	parser = argparse.ArgumentParser(description='A script to classify imdb collection using convolutional neural networks')
	return parser.parse_args()

def setLogger():
	"""
	To print logs both in console and in logs file
	"""
	logging.basicConfig(level=logging.DEBUG)
	logFormatter = logging.Formatter("%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")
	rootLogger = logging.getLogger()

	fileHandler = logging.FileHandler("{0}/{1}.log".format("logs", "logs"))
	fileHandler.setFormatter(logFormatter)
	rootLogger.addHandler(fileHandler)

	consoleHandler = logging.StreamHandler()
	consoleHandler.setFormatter(logFormatter)
	rootLogger.addHandler(consoleHandler)

class MyCallback(Callback):
	def on_train_begin(self, logs={}):
		logging.info("Training params: " + str(self.params))
		return
	def on_epoch_end(self, epoch, logs={}):
		logging.info("Epoch %d/%d\n%s" % (epoch + 1, self.params['epochs'], logs))
		return

if __name__ == "__main__":
	args = loadArgParser() # not used yet
	setLogger()
	main(args)
