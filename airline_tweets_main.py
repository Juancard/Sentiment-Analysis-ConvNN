# coding: utf-8
import sys
import os
import time
import argparse
time_format="%Y%m%d_%H%M%S"

import pandas as pd
import numpy as np
#from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.callbacks import ModelCheckpoint, Callback
from keras.preprocessing.sequence import pad_sequences

from packages.yoon_model import TextCNN
from packages import preprocess_tweets_glove, common
from packages.glove_embeddings import filterGloveEmbeddings, loadGloveEmbeddings

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

def main(config):
	logging.info("Hyperparameters: " + str(config))

	logging.info("Loading collection")	
	airline_tw = pd.read_csv(config["coll_path"], header=0)
	airline_tw = airline_tw[["text", "airline_sentiment"]]

	logging.info("Preprocessing and cleaning")
	airline_tw["text"] = airline_tw["text"].apply(preprocess_tweets_glove.tokenize)
	airline_tw["text"] = airline_tw["text"].apply(preprocess_tweets_glove.clean_str)
	#stop_words = set(stopwords.words('english')) 
	#airline_tw["text"] = airline_tw["text"].apply(lambda x:' '.join([w for w in x.split(' ') if not w in stop_words]))

	X = airline_tw["text"]
	y = pd.get_dummies(airline_tw["airline_sentiment"])

	logging.info("Setting up Training and testing sets")
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)
	logging.info("Shapes: x_train=%s, y_train=%s, x_test=%s, y_test=%s" % (
		str(X_train.shape),
		str(y_train.shape),
		str(X_test.shape),
		str(y_test.shape),
		))

	# Tokenizer
	logging.info("Fitting tokenizer")
	tokenizer = Tokenizer(num_words=None, filters='', lower=True, split=" ", char_level=False)
	tokenizer.fit_on_texts(X_train)
	# calculate max document length
	if config["sentence_length"] is None:
		config["sentence_length"] = max([len(s.split()) for s in X_train])
	# calculate vocabulary size
	if config["vocab_size"] is None:
		config["vocab_size"] = len(tokenizer.word_index) + 1
	logging.info('Max document length: %d' % config["sentence_length"])
	logging.info('Vocabulary size: %d' % config["vocab_size"])

	logging.info("Encoding data")
 	X_train = tokenizer.texts_to_sequences(X_train)
	X_train = pad_sequences(X_train, maxlen=config['sentence_length'])
	X_test = tokenizer.texts_to_sequences(X_test)
 	X_test = pad_sequences(X_test, maxlen=config['sentence_length'])

	logging.info("Generating network")
	embedding_weights = False
	if config['embed_pretrained'] == 1:
		logging.info("Loading pretrained embeddings")
		embedding_weights = filterGloveEmbeddings(
			loadGloveEmbeddings(config['embed_pretrained_path']),
			tokenizer.word_index,
			config["embed_length"],
			config["vocab_size"]
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
		embedding_train = config['embed_train'] == 1,
		classes=3
	)

	model = textCnn.model
	model.compile(loss=config['loss_func'], optimizer=config['optimizer'], metrics=['accuracy'])
	#textCnn.plot_model('plots/yoon_architecture_binary_output')

	# checkpoint
	filepath = os.path.join(config['output_path'], time.strftime(time_format) + "_" + "{epoch:02d}_{val_acc:.4f}.hdf5")
	checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
	callbacks_list = [checkpoint]

	logging.info("Training model")
	model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=config['epochs'], batch_size=config['batch_size'], callbacks=callbacks_list, verbose=2)
	model_filepath = os.path.join(
		config['output_path'],
		time.strftime(time_format) + "_" + "%depochs_%dbatchsize_%dembeddings_%dfilters_" % (config['epochs'], config['batch_size'], config['embed_length'], config['filters']) + "_".join(str(ks) for ks in config['filter_sizes']) + "filtersize" + ".h5")
	model.save(model_filepath)
	logging.info("Model saved as: " + model_filepath)

def loadArgParser():
	"""
	To handle flags used to run this script on console
	"""
	parser = argparse.ArgumentParser(description='A script to classify twitter airline collection using convolutional neural networks')
	parser.add_argument('config', nargs='+', help='.ini file with configuration data')
	return parser.parse_args()

if __name__ == "__main__":
	args = loadArgParser()
        logging = common.setLogger()
	logging.info("Starting script")
	logging.info("Loading configuration file")
	config = common.loadConfigData(args.config)
	main(config)
