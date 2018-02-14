# -*- coding: utf-8 -*-
import sys
import os
import argparse
import time
import numpy as np

time_format="%Y%m%d_%H%M%S"

# fix random seed for reproducibility
# Set only during development
seed = 7
np.random.seed(seed)

# Dependencias de módulos creados por mi para otros proyectos:
from packages.yoon_model import TextCNN
from packages.glove_embeddings import loadGloveEmbeddings,filterGloveEmbeddings
from packages import common, my_callbacks
from keras.datasets import imdb
from keras.callbacks import ModelCheckpoint, Callback
from keras.preprocessing import sequence

# Parameters for imdb dataset
INDEX_FROM = 3

def main(config, logging):
	logging.info("Hyperparameters: " + str(config))

	logging.info("Loading collection")
	(x_train, y_train), (x_test, y_test) = imdb.load_data(
		path=config["coll_path"],
		num_words=config["vocab_size"],
		skip_top=0,
		maxlen=None,
		seed=113,
		start_char=1,
		oov_char=2,
		index_from=INDEX_FROM
	)
	logging.info("Shapes: x_train=%s, y_train=%s, x_test=%s, y_test=%s" % (
		str(x_train.shape),
		str(y_train.shape),
		str(x_test.shape),
		str(y_test.shape),
		))

	logging.info("Encoding training and test sets")
	x_train = sequence.pad_sequences(x_train, maxlen=config['sentence_length'])
	x_test = sequence.pad_sequences(x_test, maxlen=config['sentence_length'])

	logging.info("Generating network")
	embedding_weights = False
	if config['embed_pretrained'] == 1:
		print "Loading pretrained embeddings"
		embedding_weights = filterGloveEmbeddings(
			loadGloveEmbeddings(config['embed_pretrained_path']),
			imdb.get_word_index(),
			config["embed_length"],
			config["vocab_size"],
			index_from=INDEX_FROM
		)
	textCnn = TextCNN(
		config['sentence_length'],
		config['embed_length'],
		config['vocab_size'],
		config['filters'],
		config['filter_sizes'],
		config['dropout_prop'],
		embedding_pretrain = config['embed_pretrained'] == 1,
		embedding_weights = embedding_weights,
		embedding_train = config['embed_train'] == 1
	)

	model = textCnn.model
	model.compile(loss=config['loss_func'], optimizer=config['optimizer'], metrics=['accuracy'])
	#textCnn.plot_model('plots/yoon_architecture_binary_output')

	# checkpoint
	filepath = os.path.join(config['output_path'], time.strftime(time_format) + "_" + "{epoch:02d}_{val_acc:.4f}.hdf5")
	checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')

	# callbacks
	interval_evaluation = my_callbacks.IntervalEvaluation(validation_data=(X_test, y_test), interval=1)
	callbacks_list = [checkpoint, interval_evaluation]

	logging.info("Training model")
	model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=config['epochs'], batch_size=config['batch_size'], callbacks=callbacks_list, verbose=2)
	model_filepath = os.path.join(
		config['output_path'],
		time.strftime(time_format) + "_" + "%depochs_%dbatchsize_%dembeddings_%dfilters_" % (config['epochs'], config['batch_size'], config['embed_length'], config['filters']) + "_".join(str(ks) for ks in config['filter_sizes']) + "filtersize" + ".h5")
	model.save(model_filepath)
	logging.info("Model saved as: " + model_filepath)

def loadArgParser():
	"""
	To handle flags used to run this script on console
	"""
	parser = argparse.ArgumentParser(description='A script to classify imdb collection using convolutional neural networks')
	parser.add_argument('config', nargs='+', help='.ini file with configuration data')
	return parser.parse_args()

if __name__ == "__main__":
	args = loadArgParser()
	logging = common.setLogger()
	logging.info("Starting script")
	logging.info("Loading configuration file")
	config = common.loadConfigData(args.config)
	main(config, logging)
