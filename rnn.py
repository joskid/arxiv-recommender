# Implements the weak supervision task of training a recurrent neural network
# to predict the LDA topic assignments for abstracts

from data_utils import *
import tensorflow as tf
import matplotlib.pyplot as plt
# suppress warnings about CPU
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class Config:
	"""
	Helper class that stores model hyperparameters and information about dataset.
	RNN objects are passed a Config() object at instantiation, so any call to
	the variables in Config should use self.config.variable_name.
	"""
	num_epochs = 10
	batch_size = 128
	num_classes = 10
	embed_size = 200
	max_length = 200
	hidden_size = 300
	learning_rate = 0.001
	dropout_rate = 0.5

class RNN():
	"""
	Class that abstracts from the TensorFlow computational graph for a RNN learning
	task. Contains methods that build the computational graph and train the RNN.
	"""
	def __init__(self, config, pretrained_embeddings):
		"""
		Basic constructor that initializes class variables.
		"""
		self.config = config
		self.pretrained_embeddings = pretrained_embeddings
		self.add_placeholders()
		self.logits = self.add_prediction_op()
		self.loss = self.add_loss_op(self.logits)
		self.train_op = self.add_training_op(self.loss)

	def add_placeholders(self):
		"""
		Creates placeholder tensors to store the input vectorized abstracts, their unpadded 
		lengths, ground truth labels, and dropout rate.
		"""
		self.abstracts_placeholder = tf.placeholder(tf.int32, shape=(None, self.config.max_length))
		self.lengths_placeholder = tf.placeholder(tf.int32, shape=(None,))
		self.labels_placeholder = tf.placeholder(tf.int32, shape=(None,))
		self.dropout_placeholder = tf.placeholder(tf.float32)

	def create_feed_dict(self, abstracts_batch, lengths_batch, labels_batch=None, dropout_rate=1):
		"""
		Helper method that creates the feed dictionary that maps from placeholder to input
		values.
		"""
		feed_dict = {}
		feed_dict[self.abstracts_placeholder] = abstracts_batch
		feed_dict[self.lengths_placeholder] = lengths_batch
		if labels_batch is not None: feed_dict[self.labels_placeholder] = labels_batch
		feed_dict[self.dropout_placeholder] = dropout_rate
		return feed_dict
	
	def add_embedding_op(self):
		"""	
		Adds the embedding layer that maps from the vectorized abstracts to word embeddings.
		"""
		embedding_tensor = tf.Variable(self.pretrained_embeddings)
		lookup = tf.nn.embedding_lookup(embedding_tensor, self.abstracts_placeholder)
		return lookup

	def add_prediction_op(self):
		"""
		Adds one layer of LSTM cells that computes a hidden state vector for each full pass of
		an abstract, and uses the hidden state vector to compute softmax logit scores for the
		classification labels (0 to 9).
		"""
		x = self.add_embedding_op()
		batch_size = tf.shape(x)[0]
		cell = tf.nn.rnn_cell.BasicLSTMCell(self.config.hidden_size)
		init_state = tf.nn.rnn_cell.LSTMStateTuple(tf.zeros(shape=(batch_size, self.config.hidden_size), dtype=tf.float64),
					  							   tf.zeros(shape=(batch_size, self.config.hidden_size), dtype=tf.float64))

		U = tf.get_variable("U", shape=(self.config.hidden_size, self.config.num_classes), dtype=tf.float64, 
											initializer = tf.contrib.layers.xavier_initializer())
		outputs, states = tf.nn.dynamic_rnn(cell, inputs=x, sequence_length=self.lengths_placeholder, initial_state=init_state, dtype=tf.float64)
		# use the hidden state (corresponding to states[1]) when calculating logits
		logits = tf.matmul(states[1], U)

		return logits

	def add_loss_op(self, logits):
		"""
		Adds the final layer that computes the cross entropy loss, assuming that class
		probabilities are calculated using a softmax function.
		"""
		loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.labels_placeholder, logits=logits))
		return loss

	def add_training_op(self, loss):
		"""
		Creates the training operation, in the form of an AdamOptimizer object that minimizes
		the cross entropy loss returned by add_loss_op().
		"""
		optimizer = tf.train.AdamOptimizer(learning_rate=self.config.learning_rate)
		return optimizer.minimize(loss)

	def train_on_batch(self, sess, abstracts_batch, lengths_batch, labels_batch):
		"""
		Runs the full computational graph on one batch of abstracts to perform one step
		of parameter updates.
		"""
		# create the feed dictionary for this batch
		feed_dict = self.create_feed_dict(abstracts_batch, lengths_batch, labels_batch, self.config.dropout_rate)
		print(sess.run([self.train_op, self.loss], feed_dict))

def preprocess_data(topics_file, labels_file, abstracts_file, embeddings_file, max_length):
	"""
	Helper function to load and preprocess data for the RNN. Returns the padded, vectorized
	abstracts, along with their unpadded lengths, the LDA labels, and word embeddings.
	"""
	# load LDA topics and abstract labels into memory (as lists)
	topics, labels = load_labels(topics_file, labels_file)
	# load tokenized abstracts and file names into memory (as lists)
	fnames, abstracts = load_abstracts(abstracts_file)
	# load pre-trained embeddings and vocabulary into memory (as arrays)
	vocab, embeddings = load_embeddings_array(embeddings_file)
	# pad abstracts, and add <NULL> token to vocabulary and word embeddings
	new_abstracts, orig_lengths, new_vocab, new_embeddings = pad_abstracts(abstracts, vocab, embeddings, max_length)
	# vectorize abstracts
	vectorized_abstracts = vectorize_abstracts(new_abstracts, new_vocab)
	return(vectorized_abstracts, orig_lengths, labels, new_embeddings)

def train(abstracts, lengths, labels, embeddings):
	"""
	Main loop that implements the training over all epochs.
	"""
	config = Config()
	rnn = RNN(config, embeddings)
	print("Initialized RNN object.")
	init = tf.global_variables_initializer()

	with tf.Session() as session:
		session.run(init)
		for abs_batch, len_batch, lab_batch in get_minibatches(abstracts, lengths, labels, config.batch_size):
			rnn.train_on_batch(session, abs_batch, len_batch, lab_batch)

if __name__ == "__main__":

	# get command line arguments
	parser = argparse.ArgumentParser()
	parser.add_argument("--lda-topics", type=str, default="lda_topics_final", help="lda_topics file")
	parser.add_argument("--lda-assignments", type=str, default="lda_assignments_final", help="lda_assignments file")
	parser.add_argument("--abs-dir-tok", type=str, default="data/abstracts_tokenized", help="Directory that stores tokenized abstracts.")
	parser.add_argument("--embeddings", type=str, default="glove/embeddings.txt", help="Path to pre-trained word embeddings.")
	parser.add_argument("--max-length", type=int, default=200, help="Maximum abstract length.")
	args = parser.parse_args()	

	abstracts, lengths, labels, embeddings = preprocess_data(args.lda_topics, args.lda_assignments, args.abs_dir_tok, args.embeddings, args.max_length)
	train(abstracts, lengths, labels, embeddings)
