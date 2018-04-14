# Implements k-nearest neightbors search using the hidden state vectors
# produced by rnn.py. 

from sklearn.neighbors import NearestNeighbors
from data_utils import load_abstracts
import argparse
import numpy as np
import time
import pickle

def load_states(states_file, num_rows):
	"""
	Loads and returns hidden states for all abstracts, when num_rows is set to None.
	"""
	start = time.time()
	print("Loading hidden states...")
	out = np.genfromtxt(states_file, max_rows=num_rows)
	print("Finished loading hidden states. Time taken: %.2f seconds." % (time.time()-start))
	return out

def get_index(query_code, codes):
	"""
	Takes in the arXiv code of the query paper and returns its index in the list of all
	arXiv codes.
	"""
	return codes.index(query_code)

def nearest_neighbors(state_vectors, query, exclude_query=True, K=10):
	"""
	Finds the indices of the k-nearest neighbors for an abstract whose state 
	vector is indexed by the argument @query. The parameter @exclude_query excludes
	the query abstract from the list of K neighbors, causing the function to return K+1
	neighbors instead.
	"""
	if exclude_query: K += 1
	knn = NearestNeighbors(n_neighbors=K, metric="euclidean")
	knn.fit(state_vectors)
	# if @query is an index for some in-corpus abstract
	if exclude_query:
		neighbors_index = knn.kneighbors(state_vectors[query:(query+1),:], return_distance=False)
		return query, neighbors_index
	# if @query is a vector for the out-of-corpus test abstract
	else:
		neighbors_index = knn.kneighbors(np.reshape(query, newshape=(1, -1)), return_distance=False)
		return neighbors_index
	
def get_titles(db_file, fnames, query, neighbors, exclude_query=True):
	"""
	Obtains titles and abstracts of nearest neighbors from the database pickle.
	@fnames are the keys required to index the dictionary stored in the database pickle.
	"""
	db = pickle.load(open(db_file, "rb"))
	# if @query is an index for some in-corpus abstract
	if exclude_query: 
		# obtain paper title for the abstract indexed by @query
		query_info = db[fnames[query]]
		query_title = query_info["title"]
		neighbors = neighbors[0][1:]
		print("======================================================================================")
		print("Getting nearest neighbors for paper titled: \n%s (%s)" % (" ".join(query_title.split()), fnames[query]))
		print("--------------------------------------------------------------------------------------")
	# if @query is a vector for the out-of-corpus test abstract
	else:
		neighbors = neighbors[0]
		print("======================================================================================")
		print("Getting nearest neighbors for paper with abstract: \n%s" % (" ".join(query.split())))
		print("--------------------------------------------------------------------------------------")
	# print out neighbors
	for i, neighbor in enumerate(neighbors):
		neighbor_info = db[fnames[neighbor]]
		neighbor_title = neighbor_info["title"]
		print("Closest neighbor no. %i:\n\t %s (%s)" % (i+1, " ".join(neighbor_title.split()), fnames[neighbor]))

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--db-name", type=str, default="db.p", help="Path to and name of database pickle.")
	parser.add_argument("--abs-dir-tok", type=str, default="data/abstracts_tokenized", help="Directory that stores tokenized abstracts.")
	parser.add_argument("--hidden-states", type=str, default="hidden_states", help="Text file that stores the hidden states output by rnn.py.")
	parser.add_argument("--num-neighbors", type=int, default=10, help="Number of nearest neighbors to find.")
	parser.add_argument("--query-index", type=int, default=None, help="Index of abstract whose nearest neighbors we want to obtain.")
	parser.add_argument("--query-code", type=str, default=None, help="Code of paper whose nearest neighbors we want to obtain.")
	parser.add_argument("--test", action="store_true", default=False, help="Run knn search on test abstract")
	parser.add_argument("--hidden-test", type=str, default="hidden_states_test", help="Hidden states for the test abstract(s)")
	parser.add_argument("--test-dir", type=str, default="data/test/", help="File path to test abstract")
	parser.add_argument("--test-abstract", type=str, default=None, help="File name of test abstract")
	args = parser.parse_args()

	# load hidden states for all abstracts
	states = load_states(args.hidden_states, num_rows=None)
	# get file names for abstracts
	fnames , _ = load_abstracts(args.abs_dir_tok)

	# if a querying by index in corpus
	if args.query_index is not None:
		query, neighbors = nearest_neighbors(states, args.query_index, K=args.num_neighbors)
		get_titles(args.db_name, fnames, query, neighbors)

	# if querying by arXiv paper code
	elif args.query_code is not None:
		query_index = get_index(args.query_code, fnames)
		query, neighbors = nearest_neighbors(states, query_index, K=args.num_neighbors)
		get_titles(args.db_name, fnames, query, neighbors)
	
	# if querying a test abstract
	elif args.test:
		test_vector = np.genfromtxt("hidden_states_test")
		assert args.test_abstract is not None, "Please enter file name of test abstract"
		with open(args.test_dir + args.test_abstract, "r") as f:
			abstract = f.read()
		neighbors = nearest_neighbors(states, test_vector, exclude_query=False, K=args.num_neighbors)
		get_titles(args.db_name, fnames, abstract, neighbors, exclude_query=False)
	
	else:
		raise ValueError("Please enter either a query index or query code.")
