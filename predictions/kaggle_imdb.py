# -*- coding: utf-8 -*-

## Prediction over kaggle test data
# DOWNLOAD COLLECTION HERE:
# https://www.kaggle.com/c/word2vec-nlp-tutorial/data

import ConfigParser
import pandas as pd
import argparse
import logging
import sys
import os
from keras.models import load_model
from keras.datasets import imdb
from keras.preprocessing.text import text_to_word_sequence
from keras.preprocessing import sequence



from models.yoon_model import TextCNN

def loadArgParser():
    parser = argparse.ArgumentParser(description='A script to make predictions with the loaded models')
    parser.add_argument("ini_filepath", help="The ini file that contains data related to configuration of this prediction")
    return parser.parse_args()

def loadIni(ini_path):
    INI_PATH = os.path.realpath(ini_path)
    Config = ConfigParser.ConfigParser()
    Config.read(ini_path)
    logging.info(ini_path)
    iniData = {}
    sections = Config.sections()
    for option in Config.options(sections[0]):
        opValue = Config.get(sections[0], option)
        iniData[option] = opValue if opValue != -1 else False;
    return iniData


def preprocess_imdb(review, word_index, vocabulary_size):
    INDEX_FROM=3
    INDEX_UNK=2
    INDEX_START=1
    # clean and tokenize
    words_list = text_to_word_sequence(review)
    # init array
    words_index_list = []
    # INDEX_START as first element in sequence (keras convention)
    words_index_list.append(INDEX_START)
    #words_index_list[0] = INDEX_START
    for word_pos in xrange(0,len(words_list)):
        word = words_list[word_pos]
        if word not in word_index:
            words_index_list.append(INDEX_UNK)
            #words_index_list[word_pos + 1] = INDEX_UNK
        else:
            words_index_list.append(word_index[word] + INDEX_FROM if word_index[word] < vocabulary_size - INDEX_FROM - 1 else INDEX_UNK)
    return words_index_list

def main(test_data_filepath, model_filepath):
    print("Starting predictions")

    print("Loading model")
    model = load_model(model_filepath)

    print "Reading test data from: " + test_data_filepath
    kaggle_test_df = pd.read_csv( test_data_filepath, header=0, delimiter="\t", quoting=3, encoding="utf-8" )

    print "Imdb dataset: Loading map from word to index"
    word_index = imdb.get_word_index()

    vocabulary_size = model.layers[1].input_dim
    sentence_length = model.layers[1].input_length

    print "Preprocessing and encoding sentences"
    preprocess_test = kaggle_test_df['review'].apply(preprocess_imdb, args=(word_index, vocabulary_size))
    x_test_predict = sequence.pad_sequences(preprocess_test, maxlen=sentence_length)
    print("Shape of dataset to predict: " + str(x_test_predict.shape))

    print "Predicting test dataset"
    model_predictions = model.predict(x_test_predict, verbose=1)
    print(model_predictions[:5])

    mod_pred_round = [int(round(i)) for i in model_predictions]
    print(mod_pred_round[:5])

    # Write the test results
    print "Writing prediction results"
    output = pd.DataFrame( data={"id":kaggle_test_df["id"], "sentiment":mod_pred_round} )
    output.to_csv("results/" + os.path.basename(model_filepath).rsplit(".")[0] + ".csv", index=False, quoting=3 )


if __name__ == "__main__":
    args = loadArgParser()
    ini_data = loadIni(args.ini_filepath)
    main(ini_data['test_data_filepath'], ini_data['model_filepath'])
