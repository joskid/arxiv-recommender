# Calls the Stanford PTBTokenizer on each abstract and writes the tokenized
# versions into one text file each.

ABS_DIR=data/abstracts
ABS_DIR_TOK=data/abstracts_tokenized
COMBINED_FILE=data/all_abstracts_tokenized
GLOVE_DIR=glove

# create directory to store tokenized abstracts
mkdir -p $ABS_DIR_TOK
# iterate over each file in ABS_DIR
for file in $ABS_DIR/*; do
	# extract file name
	fname=`basename $file`
	# call PTBTokenizer to generate newline-separated tokens
	java -cp $GLOVE_DIR/stanford-ner.jar edu.stanford.nlp.process.PTBTokenizer $file > $ABS_DIR_TOK/$fname -lowerCase
	# replace newline with white space
	tr '\n' ' ' < $ABS_DIR_TOK/$fname > $ABS_DIR_TOK/temp && mv $ABS_DIR_TOK/temp $ABS_DIR_TOK/$fname
	# remove all <unk>s from tokens
	cat $ABS_DIR_TOK/$fname | sed -e 's/<unk>/<raw_unk>/g' > $ABS_DIR_TOK/temp && mv $ABS_DIR_TOK/temp $ABS_DIR_TOK/$fname
done
# # combine all tokenized abstracts into one newline-delimited file
# paste -s $ABS_DIR_TOK/* > $COMBINED_FILE