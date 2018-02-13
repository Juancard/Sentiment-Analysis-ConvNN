import numpy as np

def loadGloveEmbeddings(path):
	"""
	Load pre trained embeddings
	"""
	# load the whole embedding into memory
	embeddings_index = dict()
	with open(path) as f:
		for line in f:
			values = line.split()
			word = values[0]
			coefs = np.asarray(values[1:], dtype='float32')
			embeddings_index[word] = coefs
	return embeddings_index

def filterGloveEmbeddings(embeddings, word_index, emb_length, vocabulary_size, index_from=1):
	embedding_matrix = np.zeros((vocabulary_size, emb_length))
	for word, i in word_index.items():
		if i <= vocabulary_size - index_from:
			embedding_vector = embeddings.get(word)
			if embedding_vector is not None:
				embedding_matrix[i + index_from - 1] = embedding_vector
	return embedding_matrix


