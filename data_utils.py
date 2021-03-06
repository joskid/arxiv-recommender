# Utility functions to load and pre-process data.

import numpy as np
import time
import os
import sys
import argparse
import glob
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
	print("Loading pre-trained GloVe word embeddings...")
	embeddings = OrderedDict()
	for line in open(fpath):
		splitLine = line.split(" ")
		token = splitLine[0]
		vector = np.array([float(value) for value in splitLine[1:]])
		embeddings[token] = vector
	print("Finished loading embeddings. Time taken: %.2f seconds."%(time.time()-start))
	return(embeddings)

def load_embeddings_array(fpath, num_embed):
	"""
	Reads the SAVE_FILE produced by the GloVe model and returns the
	vocabulary and vectors in separate arrays. 
	"""
	start = time.time()
	print("Loading pre-trained GloVe word embeddings...")
	vocab = []
	embeddings = []
	f = open(fpath)
	for i in range(num_embed):
		splitLine = f.readline().split(" ")
		vocab.append(splitLine[0])
		embeddings.append([float(value) for value in splitLine[1:]])
	print("Finished loading embeddings. Time taken: %.2f seconds."%(time.time()-start))
	return(np.array(vocab), np.array(embeddings))

def load_labels(topics_file, assignments_file):
	"""
	Reads the lda_topics and lda_assignments files generated by lda.py.
	Returns the ground truth labels for each abstract, along with the topics
	"""
	# read in LDA topics
	topics = []
	with open(topics_file) as f:
		for line in f.read().split("\n"):
			topics.append(line)
	# remove last element which is empty
	topics.pop()	
	# read in assignments
	labels = np.genfromtxt(assignments_file, dtype=int, delimiter="\n")
	print("Finished loading LDA topic labels.")
	return (topics, labels)

def load_abstracts(abs_dir_tok):
	"""
	Reads in the tokenized abstracts, stored in individual text files, and returns
	the file names and tokenized abstracts in two separate lists.
	"""
	fnames = []
	abstracts = []
	for file in glob.glob(abs_dir_tok + "/*"):
		with open(file, "r") as f:
			abstracts.append(f.read())
			fnames.append(os.path.basename(f.name))
	print("Finished loading tokenized abstracts.")
	return(fnames, abstracts)

def pad_abstracts(abstracts, vocab, embeddings, max_length):
	"""
	Pads abstracts of length less than @max_length with a <NULL> token and
	truncates abstracts of length more than @max_length. Also adds a <NULL>
	token to the vocabulary list, and also vector of zeros to the word embeddings 
	to represent the <NULL> token.
	"""
	# ensure that <NULL> token doesnt yet exist in vocabulary
	assert "<NULL>" not in vocab, "<NULL> token already exists in vocabulary. Use different null token" 
	# initialize lists to store padded abstracts and original length
	new_abstracts = []
	orig_length = []
	# loop over original list of abstracts
	for abstract in abstracts:
		# split abstracts into tokens
		tokens = abstract.split(" ")
		orig_length.append(len(tokens))
		# compute difference in lengths
		diff_len = max_length - len(tokens)
		# if abstract is too short, pad with <NULL> token
		if diff_len > 0:
			new_abstracts.append(abstract + "<NULL> "*diff_len)
		# if sentence is too long, truncate to max_length
		elif diff_len < 0: 
			new_abstracts.append(" ".join(tokens[:max_length]))
		# if correct length, don't modify
		else: 
			new_abstracts.append(abstract)
	# get depth of word embeddings
	depth = embeddings.shape[1]
	# return padded abstracts, and expanded vocabulary and word embeddings
	return(new_abstracts, np.array(orig_length), np.append(vocab, "<NULL>"), np.append(embeddings, [[0.]*depth], axis=0))

def test_pad_abstracts(abs_file, embed_file, max_length):
	"""
	Tests the pad_abstracts() function.
	"""
	# Load the abstracts in a list and embeddings in a dictionary
	_ , abstracts = load_abstracts(abs_file)
	vocab, embeddings = load_embeddings_array(embed_file)
	# run the pad_abstracts() function
	abstracts_pad, _ , _ = pad_abstracts(abstracts, max_length, vocab, embeddings)
	# check that length of all abstracts equals max_length
	for i in range(len(abstracts_pad)):
		assert len(abstracts_pad[i].split(" ")) == max_length, "Length of abstract %i is %i, not equal to %i"%(i, len(abstracts_pad[i].split(" ")), max_length)

def vectorize_abstracts(abstracts, vocab):
	"""
	Converts each token in each abstract into an integer that represents
	the token's position in the list of vocabulary.
	"""
	# convert vocab list into dictionary with token as the key and position as the value
	vocab_dict = dict(zip(vocab, list(range(len(vocab)))))
	# get index of <NULL> token
	null_index = vocab_dict["<NULL>"]
	# iterate over all abstracts and replace token with its position in vocabulary
	vectorized_abstracts = []
	for abstract in abstracts:
		vectorized_abstracts.append([vocab_dict.get(token, null_index) for token in abstract.split(" ")])
	
	return np.array(vectorized_abstracts)

def get_minibatches(abstracts, lengths, labels, batch_size, shuffle=True):
	"""
	Returns a generator that iterates over all minibatches of the training set.
	Assumes that @abstracts, @lengths, and @labels are all lists.
	"""
	# check if number of abstracts matches number of labels
	assert len(abstracts) == len(labels)
	# create indices for abstracts and shuffle if specified
	num_abstracts = len(abstracts)
	indices = np.arange(num_abstracts)
	if shuffle: np.random.shuffle(indices)
	# break up indices into slices of @batch_size each, and return generator
	for start_index in np.arange(0, num_abstracts, batch_size):
		batch_indices = indices[start_index:(start_index+batch_size)]
		yield abstracts[batch_indices], lengths[batch_indices], labels[batch_indices]

def test_get_minibatches(abs_file, topics_file, labs_file, batch_size, shuffle=True):
	"""
	Tests the get_minibatches() function.
	"""
	# Load labels and abstracts as lists
	_ , abstracts = load_abstracts(abs_file)
	_ , labels = load_labels(topics_file, labs_file)
	# run the get_minibatches() function to obtain generator
	minibatches = get_minibatches(abstracts, labels, batch_size, shuffle)
	for minibatch in minibatches:
		assert len(minibatch[0]) == batch_size
		assert len(minibatch[1]) == batch_size

class Progbar(object):
    """
    Progbar class copied from keras (https://github.com/fchollet/keras/)
    Displays a progress bar.
    # Arguments
        target: Total number of steps expected.
        interval: Minimum visual progress update interval (in seconds).
    """

    def __init__(self, target, width=30, verbose=1):
        self.width = width
        self.target = target
        self.sum_values = {}
        self.unique_values = []
        self.start = time.time()
        self.total_width = 0
        self.seen_so_far = 0
        self.verbose = verbose

    def update(self, current, values=None, exact=None):
        """
        Updates the progress bar.
        # Arguments
            current: Index of current step.
            values: List of tuples (name, value_for_last_step).
                The progress bar will display averages for these values.
            exact: List of tuples (name, value_for_last_step).
                The progress bar will display these values directly.
        """
        values = values or []
        exact = exact or []

        for k, v in values:
            if k not in self.sum_values:
                self.sum_values[k] = [v * (current - self.seen_so_far), current - self.seen_so_far]
                self.unique_values.append(k)
            else:
                self.sum_values[k][0] += v * (current - self.seen_so_far)
                self.sum_values[k][1] += (current - self.seen_so_far)
        for k, v in exact:
            if k not in self.sum_values:
                self.unique_values.append(k)
            self.sum_values[k] = [v, 1]
        self.seen_so_far = current

        now = time.time()
        if self.verbose == 1:
            prev_total_width = self.total_width
            sys.stdout.write("\b" * prev_total_width)
            sys.stdout.write("\r")

            numdigits = int(np.floor(np.log10(self.target))) + 1
            barstr = '%%%dd/%%%dd [' % (numdigits, numdigits)
            bar = barstr % (current, self.target)
            prog = float(current)/self.target
            prog_width = int(self.width*prog)
            if prog_width > 0:
                bar += ('='*(prog_width-1))
                if current < self.target:
                    bar += '>'
                else:
                    bar += '='
            bar += ('.'*(self.width-prog_width))
            bar += ']'
            sys.stdout.write(bar)
            self.total_width = len(bar)

            if current:
                time_per_unit = (now - self.start) / current
            else:
                time_per_unit = 0
            eta = time_per_unit*(self.target - current)
            info = ''
            if current < self.target:
                info += ' - ETA: %ds' % eta
            else:
                info += ' - %ds' % (now - self.start)
            for k in self.unique_values:
                if isinstance(self.sum_values[k], list):
                    info += ' - %s: %.4f' % (k, self.sum_values[k][0] / max(1, self.sum_values[k][1]))
                else:
                    info += ' - %s: %s' % (k, self.sum_values[k])

            self.total_width += len(info)
            if prev_total_width > self.total_width:
                info += ((prev_total_width-self.total_width) * " ")

            sys.stdout.write(info)
            sys.stdout.flush()

            if current >= self.target:
                sys.stdout.write("\n")

        if self.verbose == 2:
            if current >= self.target:
                info = '%ds' % (now - self.start)
                for k in self.unique_values:
                    info += ' - %s: %.4f' % (k, self.sum_values[k][0] / max(1, self.sum_values[k][1]))
                sys.stdout.write(info + "\n")

    def add(self, n, values=None):
        self.update(self.seen_so_far+n, values)

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--lda-topics", type=str, default="lda_topics", help="lda_topics file")
	parser.add_argument("--lda-assignments", type=str, default="lda_assignments", help="lda_assignments file")
	parser.add_argument("--abs-dir-tok", type=str, default="data/abstracts_tokenized", help="Directory that stores tokenized abstracts.")
	parser.add_argument("--embeddings", type=str, default="glove/embeddings.txt", help="Path to pre-trained word embeddings.")
	args = parser.parse_args()	

	# test_get_minibatches(args.abs_dir_tok, args.lda_topics, args.lda_assignments, batch_size=1000, shuffle=True)
	# test_pad_abstracts(args.abs_dir_tok, args.embeddings, max_length=300)
	vocab, embeddings = load_embeddings_array(args.embeddings, num_embed=200000)
	print(embeddings.shape)