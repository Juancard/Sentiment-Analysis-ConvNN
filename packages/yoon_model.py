# -*- coding: utf-8 -*-

from keras.layers import Convolution1D, MaxPooling1D, Flatten, Dropout, Input, Dense
from keras.layers.embeddings import Embedding
from keras.layers.merge import concatenate
from keras.models import Model
from keras.utils.vis_utils import plot_model

class TextCNN(object):
    """
    A CNN for text classification based on K. Yoon paper (2014)
    Uses an embedding layer, followed by a convolutional layer
    of many different filters and kernel sizes, max-pooling on each
    and finally sigmoid layer.

    Parameters:
    sentence_length => int, length of a sentence (all sentences must have same length)
    embedding_length => int, size of vector embedding
    filters => number of filters on each filter size.
    filter_sizes => list of ints, number of words to be convoluted
    dropout => proportion of elements to be wiped out
    """
    def __init__(self,
        sentence_length,
        embedding_length,
        vocabulary_size,
        filters,
        filter_sizes,
        dropout,
        embedding_pretrain=False,
        embedding_weights=False,
        embedding_train=True,
	classes=2
        ):
        # Input layer
        # Receives the sentence
        self.input_layer = Input(
            shape=(sentence_length,),
            dtype='int32', name='sentence')

        # Embedding layer
        # each word on setence with its corresponding embedding vector.
        if embedding_pretrain:
            # create the embedding layer with pre trained embeddings
            self.embedding = Embedding(
                vocabulary_size,
                embedding_length,
                weights=[embedding_weights],
                input_length=sentence_length,
                name="embedding",
                trainable=embedding_train
            )(self.input_layer)
        else:
            self.embedding = Embedding(
                vocabulary_size,
                embedding_length,
                input_length=sentence_length,
                name="embedding",
                trainable=embedding_train
            )(self.input_layer)

        # COnvolutional layers
        # One for each filter size
        # Each conv. layer with a max pooling
        self.convolutional_layers = []
        self.maxpooling_layers = []
        for i, kernel_size in enumerate(filter_sizes):
            name = str(kernel_size) + "ks"
            conv = Convolution1D(
                filters=filters,
                kernel_size=kernel_size,
                activation='relu',
                name="conv_" + name
            )(self.embedding)
            maxPooling = MaxPooling1D(
                pool_size= sentence_length - kernel_size + 1,
                name="maxpool_" + name
            )(conv)
            self.convolutional_layers.append(conv)
            self.maxpooling_layers.append(maxPooling)
        # Merge outputs of each conv-maxpooling layer
        self.merge = concatenate(self.maxpooling_layers, name="concatenation")
        # Flatten output
        self.flat = Flatten(name="flatten_layer")(self.merge)
        # Dropout layer
        # to prevent overfitting
        self.drop = Dropout(dropout, name="dropout_%.2f" % dropout)(self.flat)
        # Output layer
        # Final binary classification
	if classes == 2:
        	self.output = Dense(1, activation='sigmoid')(self.drop)
        else:
		self.output = Dense(classes, activation='softmax')(self.drop)
	self.model = Model(inputs=self.input_layer, outputs=self.output, name="binary_output")

    def plot_model(self, filepath):
        plot_model(self.model, show_shapes=True, to_file=filepath)
