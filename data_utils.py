# Utility functions to load and pre-process data.

import numpy as np
import time
from collections import OrderedDict
from pprint import pprint

def load_vocab(fpath):
	"""
	Reads the VOCAB_FILE produced by the PTBTokenizer and returns all 
	(token, count) pairs in a numpy array.
	"""
	return np.genfromtxt(fpath, dtype=("U25, int"), comments=None, delimiter=" ")

def load_embeddings(fpath):
	"""
	Reads the SAVE_FILE produced by the GloVe model and returns the
	(token, vector) pairs in an ordered dictionary.
	"""
	start = time.time()
	print("Loading pre-trained GloVe word embeddings.")
	embeddings = OrderedDict()
	for line in open(fpath):
		splitLine = line.split(" ")
		token = splitLine[0]
		vector = np.array([float(value) for value in splitLine[1:]])
		embeddings[token] = vector
	print("Finished loading embeddings. Time taken: %.2f seconds."%(time.time()-start))
	return(embeddings)

if __name__ == "__main__":
	vocab = load_vocab("glove/vocab.txt")
	embeddings = load_embeddings("glove/embeddings.txt")