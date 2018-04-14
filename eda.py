# Performs basic exploratory data analysis on the generated tokens and word embeddings.

import string
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from data_utils import *
from nltk.corpus import stopwords

def get_top_fifty(token_and_counts, K=50, print_mat=False):
	"""
	Takes in a vocabulary matrix that stores (token, count) pairs and returns
	the top K (50 by default) most frequent tokens that are not common stop 
	words, punctuations, or numbers, along with their counts.
	"""
	top50 = []
	index = 0
	for token, count in iter(token_and_counts):
		if token.lower() not in stopwords.words('english'):
			if token not in string.punctuation:
				if not token.isdigit():
					top50.append((index, token, count))
					index += 1
		if index == K:
			out = np.array(top50, dtype=("int, U25, int"))
			if print_mat: print(out)
			return out

def plot_counts(top_fifty, indices):
	"""
	Takes as input the top fifty (token, count) pairs, and visualizes the counts
	for the tokens that are indexed by the ndarray of indices.
	"""
	plot_df = pd.DataFrame(data=top_fifty[indices])	
	plot_df.columns = ["rank", "token", "count"]
	ax = plot_df.plot.bar(x="token", y="count", color='blue', legend=None, figsize=(8, 3))
	ax.set_xlabel("Token", fontsize=14); ax.set_ylabel("Count", fontsize=12)
	plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
	plt.gcf().tight_layout()
	plt.show()

def visualize_words(embed_dict, num_words, use_prev):
	"""
	Performs t-SNE dimensionality reduction on the word embeddings of the @num_words
	most frequent tokens to visualize word similarities on a 2-D scatterplot. Setting
	@use_prev to true loads in previously calculate t-SNE vectors.
	"""
	# convert dictionary of word vectors into numpy array
	embeddings = np.array(list(embed_dict.values()))
	# load previously saved t-SNE results
	if use_prev: 
		lowDim = np.genfromtxt("tsne", dtype=float, delimiter=" ")
		print("Loaded previously calculated t-SNE vectors.")
	# otherwise, run t-SNE and write new embeddings to text file		
	else:
		start = time.time()
		print("Performing t-SNE dimensionality reduction on word embeddings.")
		lowDim = TSNE(n_components=2, n_iter=2000, early_exaggeration=12.0).fit_transform(embeddings[:num_words])
		print("Finished t-SNE. Time taken: %.2f seconds."%(time.time()-start))
		np.savetxt("tsne", lowDim, delimiter=" ")
	assert lowDim.shape[0] == num_words, "Loaded t-SNE vectors calculated for a different number of words."
	# create scatter plot of new low-dimensional vectors
	plt.figure(figsize=(12, 8))
	plt.scatter(lowDim[:, 0], lowDim[:, 1], marker="o")
	plt.xlabel("x1", fontsize=14); plt.ylabel("x2", fontsize=14)
	# annotate plotted vectors using token names
	for i, word in enumerate(list(embed_dict.keys())[:num_words]):
		plt.annotate(word, xy = (lowDim[i, 0], lowDim[i, 1]), xytext=(3, 3),
					 textcoords="offset points", fontsize=17)
	plt.show()

if __name__ == "__main__":
	# load vocabulary and counts
	print("Loading vocabulary and counts...")
	vocab = load_vocab("glove/vocab.txt")
	# obtain fifty most frequent tokens
	top50 = get_top_fifty(vocab, print_mat=False)
	# plot frequent tokens and counts
	plot_counts(top50, np.array([11, 12, 13, 15, 26, 27, 28, 29, 33, 35]))
	# load pre-trained GloVe word embeddings
	embeddings = load_embeddings("glove/embeddings.txt")
	# visualize word similarities using t-SNE
	visualize_words(embeddings, num_words=1000, use_prev=False)