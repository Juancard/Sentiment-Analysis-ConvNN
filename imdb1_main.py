# -*- coding: utf-8 -*-
import sys
import os
import logging
import argparse
import configparser 
import time
import numpy as np
time_format="%Y%m%d_%H%M%S"

# fix random seed for reproducibility
# Set only during development
seed = 7
np.random.seed(seed)

# Dependencias de módulos creados por mi para otros proyectos:
from models.yoon_model import TextCNN
from keras.datasets import imdb
from keras.callbacks import ModelCheckpoint, Callback
from keras.preprocessing import sequence

def main(config):
	logging.info("Hyperparameters: " + str(config))
	logging.info("Generating network")
	textCnn = TextCNN(config['sentence_length'],
		config['embed_length'],
		config['vocab_size'],
		config['filters'],
		config['filter_sizes'],
		config['dropout_prop'])
	model = textCnn.model
	model.compile(loss=config['loss_func'], optimizer=config['optimizer'], metrics=['accuracy'])
	#textCnn.plot_model('plots/yoon_architecture_binary_output')
	logging.info("Loading collection")
	(x_train, y_train), (x_test, y_test) = imdb.load_data(
		path=config["coll_path"],
		num_words=config["vocab_size"],
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
	x_train = sequence.pad_sequences(x_train, maxlen=config['sentence_length'])
	x_test = sequence.pad_sequences(x_test, maxlen=config['sentence_length'])

	# checkpoint
	filepath = os.path.join(config['output_path'], time.strftime(time_format) + "_" + "{epoch:02d}_{val_acc:.4f}.hdf5")
	checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
	# logs during training
	callback1 = MyCallback()
	callbacks_list = [checkpoint, callback1]

	print "Training model"
	model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=config['epochs'], batch_size=config['batch_size'], callbacks=callbacks_list, verbose=0)
	model_filepath = os.path.join(
		config['output_path'], 
		time.strftime(time_format) + "_" + "%depochs_%dbatchsize_%dembeddings_%dfilters_" % (config['epochs'], config['batch_size'], config['embed_length'], config['filters']) + "_".join(str(ks) for ks in config['filter_sizes']) + "filtersize" + ".h5")
	model.save(model_filepath)
	print "Model saved as: " + model_filepath

def loadArgParser():
	"""
	To handle flags used to run this script on console
	"""
	parser = argparse.ArgumentParser(description='A script to classify imdb collection using convolutional neural networks')
	parser.add_argument('config', nargs='+', help='.ini file with configuration data')
	return parser.parse_args()

def setLogger():
	"""
	To print logs
	"""
	logging.basicConfig(level=logging.DEBUG)
	logFormatter = logging.Formatter("%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")
def loadConfigData(iniFilePath):
	"""
	to load configuration data from .ini file
	"""
	config = configparser.ConfigParser()
	config.read(args.config)
	config = {
		'coll_path': config['COLLECTION']['PATH'],
		'vocab_size': int(config['MODEL']['VOCABULARY_SIZE']),
		'sentence_length': int(config['MODEL']['SENTENCE_LENGTH']),
		'embed_length': int(config['MODEL']['EMBEDDING_LENGTH']),
		'filters': int(config['MODEL']['FILTERS']),
		'filter_sizes': [int(fs) for fs in config['MODEL']['FILTER_SIZES'].split(',')],
		'epochs': int(config['FITTING']['EPOCHS']),
		'batch_size': int(config['FITTING']['BATCH_SIZE']),
		'dropout_prop': float(config['MODEL']['DROPOUT']),
		'loss_func': config['MODEL']['LOSS_FUNCTION'],
		'optimizer': config['MODEL']['OPTIMIZER'],
		'output_path': config['OUTPUT']['PATH']
	}
	return config

class MyCallback(Callback):
	def on_train_begin(self, logs={}):
		logging.info("Training params: " + str(self.params))
		return
	def on_epoch_end(self, epoch, logs={}):
		logging.info("Epoch %d/%d\n%s" % (epoch + 1, self.params['epochs'], logs))
		return

if __name__ == "__main__":
	args = loadArgParser()
        setLogger()
	logging.info("Starting script")
	logging.info("Loading configuration file")
	config = loadConfigData(args)
	main(config)
