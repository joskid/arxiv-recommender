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

def nearest_neighbors(state_vectors, center, exclude_center=True, K=10):
	"""
	Finds the indices of the k-nearest neighbors for an abstract whose state 
	vector is indexed by the argument @center. The parameter @exclude_center excludes
	the center abstract from the list of K neighbors, causing the function to return K+1
	neighbors instead.
	"""
	if exclude_center: K += 1
	knn = NearestNeighbors(n_neighbors=K, metric="euclidean")
	knn.fit(state_vectors)
	neighbors_index = knn.kneighbors(state_vectors[center:(center+1),:], return_distance=False)
	return center, neighbors_index

def get_titles(db_file, fnames, center, neighbors, exclude_center=True):
	"""
	Obtains titles and abstracts of nearest neighbors from the database pickle.
	@fnames are the keys required to index the dictionary stored in the database pickle.
	"""
	db = pickle.load(open(db_file, "rb"))
	# obtain paper title for the abstract indexed by @center
	center_info = db[fnames[center]]
	center_title = center_info["title"]
	if exclude_center: 
		neighbors = neighbors[0][1:]
	else:
		neighbors = neighbors[0]
	print("======================================================================================")
	print("Getting nearest neighbors for paper titled: \n%s (%s)" % (" ".join(center_title.split()), fnames[center]))
	print("--------------------------------------------------------------------------------------")
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
	parser.add_argument("--center-index", type=int, help="Index of abstract whose nearest neighbors we want to obtain.")
	args = parser.parse_args()

	# load hidden states for all abstracts
	states = load_states(args.hidden_states, num_rows=None)
	# find indices of nearest neighbors for a specified abstract
	center, neighbors = nearest_neighbors(states, args.center_index, K=args.num_neighbors)
	# get file names for abstracts
	fnames , _ = load_abstracts(args.abs_dir_tok)
	get_titles(args.db_name, fnames, center, neighbors)
