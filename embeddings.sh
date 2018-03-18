# Converts a corpus of arxiv papers into white-space-separated tokens,
# using the Stanford PTBTokenizer. Then, trains GloVe word embeddings
# on the tokens, using Stanford's GloVe model.

DATA_DIR=data/txt
GLOVE_DIR=glove
TOKENS_FILE=tokens.txt
VOCAB_FILE=vocab.txt
COOCCURRENCE_FILE=cooccurrence.bin
COOCCURRENCE_SHUF_FILE=cooccurrence.shuf.bin
SAVE_FILE=embeddings
NUM_THREADS=8
X_MAX=10
MAX_ITER=15
BINARY=2
VECTOR_SIZE=200

VOCAB_MIN_COUNT=5
VERBOSE=2
MEMORY=4.0
WINDOW_SIZE=15

# use PTBTokenizer to generate newline-separated tokens
echo "$ java -cp $GLOVE_DIR/stanford-ner.jar edu.stanford.nlp.process.PTBTokenizer data/txt/* > $GLOVE_DIR/raw_tokens.txt"
java -cp $GLOVE_DIR/stanford-ner.jar edu.stanford.nlp.process.PTBTokenizer $DATA_DIR/* > $GLOVE_DIR/raw_tokens.txt -lowerCase
# replace newline with white space
echo "$ tr '\n' ' ' < $GLOVE_DIR/raw_tokens.txt > $GLOVE_DIR/raw_tokens2.txt"
tr '\n' ' ' < $GLOVE_DIR/raw_tokens.txt > $GLOVE_DIR/raw_tokens2.txt
# remove all <unk>s from tokens
echo "$ tr -d '<unk>' < $GLOVE_DIR/raw_tokens2.txt > $GLOVE_DIR/$TOKENS_FILE"
cat $GLOVE_DIR/raw_tokens2.txt | sed -e 's/<unk>/<raw_unk>/g' > $GLOVE_DIR/$TOKENS_FILE
# print first 256 bytes of TOKENS_FILE
echo "$ head -c 256 $GLOVE_DIR/$TOKENS_FILE"
head -c 256 $GLOVE_DIR/$TOKENS_FILE
# delete raw token files
echo "$ rm $GLOVE_DIR/raw_tokens.txt && rm $GLOVE_DIR/raw_tokens2.txt"
rm $GLOVE_DIR/raw_tokens.txt && rm $GLOVE_DIR/raw_tokens2.txt

# run the vocab_count tool to obtain the VOCAB_FILE
echo "$ $GLOVE_DIR/vocab_count -min-count $VOCAB_MIN_COUNT -verbose $VERBOSE < $TOKENS_FILE > $VOCAB_FILE "
$GLOVE_DIR/vocab_count -min-count $VOCAB_MIN_COUNT -verbose $VERBOSE < $GLOVE_DIR/$TOKENS_FILE > $GLOVE_DIR/$VOCAB_FILE 
# run the cooccur tool to construct word-word coocurrence statistics
echo "$ $GLOVE_DIR/cooccur -memory $MEMORY -vocab-file $GLOVE_DIR/$VOCAB_FILE -verbose $VERBOSE -window-size $WINDOW_SIZE < $GLOVE_DIR/$TOKENS_FILE > $GLOVE_DIR/$COOCCURRENCE_FILE"
$GLOVE_DIR/cooccur -memory $MEMORY -vocab-file $GLOVE_DIR/$VOCAB_FILE -verbose $VERBOSE -window-size $WINDOW_SIZE < $GLOVE_DIR/$TOKENS_FILE > $GLOVE_DIR/$COOCCURRENCE_FILE
# run the shuffle tool to shuffle the binary file of cooccurrence statistics
echo "$ $GLOVE_DIR/shuffle -memory $MEMORY -verbose $VERBOSE < $GLOVE_DIR/$COOCCURRENCE_FILE > $GLOVE_DIR/$COOCCURRENCE_SHUF_FILE"
$GLOVE_DIR/shuffle -memory $MEMORY -verbose $VERBOSE < $GLOVE_DIR/$COOCCURRENCE_FILE > $GLOVE_DIR/$COOCCURRENCE_SHUF_FILE
# run the glove tool to train the GloVe model on the cooccurrence data
echo "$ $GLOVE_DIR/glove -save-file $GLOVE_DIR/$SAVE_FILE -threads $NUM_THREADS -input-file $GLOVE_DIR/$COOCCURRENCE_SHUF_FILE -x-max $X_MAX -iter $MAX_ITER -vector-size $VECTOR_SIZE -binary $BINARY -vocab-file $GLOVE_DIR/$VOCAB_FILE -verbose $VERBOSE"
$GLOVE_DIR/glove -save-file $GLOVE_DIR/$SAVE_FILE -threads $NUM_THREADS -input-file $GLOVE_DIR/$COOCCURRENCE_SHUF_FILE -x-max $X_MAX -iter $MAX_ITER -vector-size $VECTOR_SIZE -binary $BINARY -vocab-file $GLOVE_DIR/$VOCAB_FILE -verbose $VERBOSE

# # evaluate word vectors
# echo "$ python $GLOVE_DIR/eval/evaluate.py -vocab_file --$GLOVE_DIR/$VOCAB_FILE -vectors_file --$GLOVE_DIR/$embeddings"
# python $GLOVE_DIR/eval/evaluate.py --vocab_file $GLOVE_DIR/$VOCAB_FILE --vectors_file $GLOVE_DIR/$SAVE_FILE