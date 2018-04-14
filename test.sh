# Calls the Stanford PTBTokenizer on each test abstract and writes the tokenized
# versions into one text file each.

ABS_DIR=data/test
GLOVE_DIR=glove

# create directory to store tokenized abstracts
mkdir -p $ABS_DIR/tokenized
# iterate over each file in ABS_DIR
for file in $ABS_DIR/*; do
	# extract file name
	fname=`basename $file`
	# if encountered the "tokenized" folder, skip it
	if [ "$fname" == "tokenized" ]
		then
		continue
	fi
	# call PTBTokenizer to generate newline-separated tokens
	java -cp $GLOVE_DIR/stanford-ner.jar edu.stanford.nlp.process.PTBTokenizer $file > $ABS_DIR/tokenized/$fname -lowerCase
	# replace newline with white space
	tr '\n' ' ' < $ABS_DIR/tokenized/$fname > $ABS_DIR/tokenized/temp && mv $ABS_DIR/tokenized/temp $ABS_DIR/tokenized/$fname
	# remove all <unk>s from tokens
	cat $ABS_DIR/tokenized/$fname | sed -e 's/<unk>/<raw_unk>/g' > $ABS_DIR/tokenized/temp && mv $ABS_DIR/tokenized/temp $ABS_DIR/tokenized/$fname
	# remove incorrectly created "tokenized" text file if it exists
	if [ -f "$ABS_DIR/tokenized/tokenized" ]
		then
		rm $ABS_DIR/tokenized/tokenized
	fi
done