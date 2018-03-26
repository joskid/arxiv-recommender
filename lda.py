# Perform LDA on the abstracts of the downloaded papers to obtain weak
# supervision signal for the RNN.

import pickle
import string
from nltk.corpus import stopwords
from gensim import corpora
from gensim.models.ldamodel import LdaModel
from data_utils import *

def tokenize_abstracts(db, abs_dir, abs_dir_tok):
	"""
	Takes in the database pickle and returns a list of tokenized abstracts, along with
	the list of corresponding file (academic paper) names.
	"""
	# first, save each abstract as a text file in the specified directory
	if not os.path.exists(abs_dir):
		os.makedirs(abs_dir)
		for key, value in db.items():
			with open(abs_dir + key, "w+") as f:
				f.write(value["summary"])
	# next, run an external shell script to tokenize all abstracts, if not already done
	if not os.path.exists(abs_dir_tok):
		start = time.time()
		print("Running abstracts.sh ... (this will take a while)")
		os.system("./abstracts.sh")
		print("Created tokenized abstracts in %s. Time taken: %.2f seconds."%(abs_dir_tok, time.time()-start))
	# load tokenized abstracts into memory as a list
	return load_abstracts(abs_dir_tok)

def create_corpus(tokenized_abstracts):
	"""
	Creates a gensim corpus from a list of tokenized abstracts
	"""
	start = time.time()
	print("Creating corpus...")
	# remove stop words and digits from abstracts
	custom_stopWords = ["-rrb-", "-lrb-", "-rcb-", "-lcb-", ""]
	stopList = stopwords.words("english") + list(string.punctuation) + custom_stopWords
	abstracts = [[token for token in abstract.split(" ") if token not in stopList and not token.isdigit()] 
				  for abstract in tokenized_abstracts]
	# convert to bag of words representation
	dictionary = corpora.Dictionary(abstracts)
	# create gensim corpus of abstracts
	corpus = [dictionary.doc2bow(abstract) for abstract in abstracts]
	print("Finished creating corpus. Time taken: %.2f"%(time.time()-start))
	return(corpus, dictionary)

def LDA(corpus, dictionary, numTopics):
	"""
	Performs LDA on a list of abstracts represented as a gensim corpus. 
	Returns both the fitted topics and the assigned topic for each abstract
	"""
	print("Fitting LDA model...")
	lda = LdaModel(corpus, num_topics=numTopics, id2word=dictionary)
	# retrieve assigned topics for abstracts in corpus
	lda_corpus = lda[corpus]
	# iterate over each assignment and choose dominant topic 
	assigned_topics = []
	for assignment in lda_corpus:
		# if assigned only one topic, retrieve that topic
		if len(assignment) == 1:
			assigned_topics.append(assignment.pop()[0])
		# if assigned multiple topics, retrieve topic with highest probability
		else:
			main_topic = max(assignment, key=lambda item:item[1])
			assigned_topics.append(main_topic[0])
	return(lda.show_topics(), assigned_topics)

if __name__ == "__main__":
	
	# get command line arguments
	parser = argparse.ArgumentParser()
	parser.add_argument("--db-name", type=str, default="db.p", help="Path to and name of database pickle.")
	parser.add_argument("--abs-dir", type=str, default="data/abstracts/", help="Directory to store extracted abstracts in.")
	parser.add_argument("--abs-dir-tok", type=str, default="data/abstracts_tokenized", help="Directory that stores tokenized abstracts.")
	parser.add_argument("--num-topics", type=int, default=10, help="Number of LDA topics.")
	args = parser.parse_args()
	
	# load existing database into memory
	db = pickle.load(open(args.db_name, "rb"))
	# obtain list of tokenized abstracts along with filenames
	fnames, abstracts = tokenize_abstracts(db, args.abs_dir, args.abs_dir_tok)
	# create gensim corpus for LDA modelling
	corpus, bow = create_corpus(abstracts)
	# obtain fitted topics and assigned topic for each abstract
	topics, assignments = LDA(corpus, bow, args.num_topics)
	# write to text files
	with open("lda_topics", "w+") as f:
		for _, topic in topics:
			f.write("%s\n" % str(topic))
	with open("lda_assignments", "w+") as f:
		for assignment in assignments:
			f.write("%s\n" % str(assignment))
	print("Finished writing topics and assignments to text files.")