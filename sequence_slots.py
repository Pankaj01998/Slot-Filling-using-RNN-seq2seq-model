import csv
import numpy as np
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords 
from keras.utils import to_categorical
from numpy import array
from numpy import argmax
from numpy import array_equal
from numpy import newaxis
from keras.models import Model
from keras.layers import Input
from keras.layers import LSTM
from keras.layers import Dense


#reading from fin_train_slots.csv for reading slots and queries
filename = "fin_train_slots.csv"
fields = []
rows = []
with open(filename, 'r') as csvfile:
	csvreader = csv.reader(csvfile)
	fields = csvreader.next()
	rows.append(fields)
	for row in csvreader:
		rows.append(row)

#storing query in dataX and slots in dataY
dataX = []
dataY = []
for lst in rows:
	dataX.append(lst[0])
	dataY.append(lst[1])

#lemmatizing the words and removing stop words from queries and storing in new_dataX
lemmatizer = WordNetLemmatizer()
new_dataX = []
stop_words = set(stopwords.words('english')) 
distinct_words = set() #contains distinct words from query text
test_sen = [] #contains sentences for testing
counter = 0
for sen in dataX:
	if counter >= 4500:
		test_sen.append(sen)
	word_tokens = word_tokenize(sen)
	lem_sen = " "
	for word in word_tokens:
		tok = lemmatizer.lemmatize(word)
		tok = tok.lower()
		distinct_words.add(tok)
		lem_sen = lem_sen + " " + tok
	new_dataX.append(lem_sen)
	counter = counter + 1

#adding a word for padding
distinct_words.add("#####")
n_features = len(distinct_words) #stores no of features

#converts feature to integer
features_to_int = dict((c, i) for i, c in enumerate(list(distinct_words)))


X1 = list() #contains input to model
X1_test = list() #contains output for testing
inp_sen_len = set() #stores len of input sentences

#now we will find length of input sentences
for sen in new_dataX:
	word_tokens = word_tokenize(sen)
	sen_int = []
	for word in word_tokens:
		sen_int.append(features_to_int[word])
	inp_sen_len.add(len(sen_int))
	
padg = features_to_int["#####"]
eos = features_to_int["eos"]
max_len = max(list(inp_sen_len)) #gives length of longest sentences


counter = 0
#first converting all sentences to eual length of words by padding and then converting to integers using word2vec
for sen in new_dataX:
	word_tokens = word_tokenize(sen)
	sen_int = []
	#converting sentences to integer
	for word in word_tokens:
		sen_int.append(features_to_int[word])
	#padding the sentences to equal length
	if len(sen_int) != max_len:
		sen_int[len(sen_int)-1] = padg
		for x in xrange(len(sen_int)+1,max_len):
			sen_int.append(padg)
		sen_int.append(eos)
	#one hot encoding the sentences
	sen_encoded = to_categorical([sen_int], num_classes = n_features)
	#dataset for training
	if counter < 4500:
		X1.append(sen_encoded)
	#dataset for testing
	else:
		X1_test.append(sen_encoded)
	counter = counter + 1

#contining distinct words from slots text
dist_wor_out = set()

#adding word to dist_wor_out except "O" because it is serving no purpose. I am taking slot into consideration from slot text
for sen in dataY:
	word_tokens = word_tokenize(sen)
	for word in word_tokens:
		if word != "O":
			dist_wor_out.add(word)

#word for padding slot 
dist_wor_out.add("##############") 
#word for time step
dist_wor_out.add("#####")
#contains number of features for output
no_out = len(dist_wor_out)
#converts slots in dist_wor_out to integer
slot_int = dict((c, i) for i, c in enumerate(list(dist_wor_out)))
#converts integer to slots
int_slot = dict((i, c) for i, c in enumerate(list(dist_wor_out)))


out_padg = slot_int["#####"]
start_timst = slot_int["##############"]

y = list() #dataset for training as output
y_test = list() #dataset for testing as output
y_sen = [] #sentences fro testing
X2 = list() #dataset for target sequence as input to seq2seq model for training 
X2_test = list() #dataset for target sequence as input to seq2seq model for testing

#contains length of output ssentences
out_sen_len = set()

#filling out_sen_len
for sen in dataY:
	word_tokens = word_tokenize(sen)
	sen_int = []
	for word in word_tokens:
		if word != "O":
			sen_int.append(slot_int[word])
	out_sen_len.add(len(sen_int))

max_out_len = max(list(out_sen_len)) #contains length of longest output sentence


cn_op = 0;
#first converting all sentences to eual length of words by padding and then converting to integers using word2vec
for sen in dataY:
	word_tokens = word_tokenize(sen)
	sen_int = [] #stroing original output
	sen_time = [] #storing target sequence
	new_sen = "" #new sentence without "O"
	for word in word_tokens:
		if word != "O":
			sen_int.append(slot_int[word])
			new_sen = new_sen + word
	#padding the sentences of equal length
	if len(sen_int) != max_out_len:
		for x in xrange(len(sen_int)+1,max_out_len):
			sen_int.append(out_padg)
		sen_int.append(out_padg)
	sen_time = [start_timst] + sen_int[:-1] #target sequence with arbitrary starting feature
	#one hot encoding a sentence 
	sen_int = to_categorical([sen_int], num_classes = no_out)
	#one hot encoding a sentence
	sen_time = to_categorical([sen_time], num_classes = no_out)
	#dataset for training as output and target sequence
	if cn_op < 4500:
		y.append(sen_int)
		X2.append(sen_time)
	#dataset for training as output and target sequence
	else:
		y_sen.append(new_sen)
		y_test.append(sen_int)
		X2_test.append(sen_time)
	cn_op = cn_op + 1


# returns train, inference_encoder and inference_decoder models
def define_models(n_input, n_output, n_units):
	# define training encoder
	encoder_inputs = Input(shape=(None, n_input))
	encoder = LSTM(n_units, return_state=True)
	encoder_outputs, state_h, state_c = encoder(encoder_inputs)
	encoder_states = [state_h, state_c]
	# define training decoder
	decoder_inputs = Input(shape=(None, n_output))
	decoder_lstm = LSTM(n_units, return_sequences=True, return_state=True)
	decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
	decoder_dense = Dense(n_output, activation='softmax')
	decoder_outputs = decoder_dense(decoder_outputs)
	model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
	# define inference encoder
	encoder_model = Model(encoder_inputs, encoder_states)
	# define inference decoder
	decoder_state_input_h = Input(shape=(n_units,))
	decoder_state_input_c = Input(shape=(n_units,))
	decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
	decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)
	decoder_states = [state_h, state_c]
	decoder_outputs = decoder_dense(decoder_outputs)
	decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)
	# return all models
	return model, encoder_model, decoder_model


# generate target given source 
def predict_sequence(infenc, infdec, source, n_steps, cardinality):
	# encode
	state = infenc.predict(source)
	# start of sequence input
	target_seq = array([0.0 for _ in range(cardinality)]).reshape(1, 1, cardinality)
	# collect predictions
	output = list()
	for t in range(n_steps):
		# predict next word
		yhat, h, c = infdec.predict([target_seq] + state)
		# store prediction
		output.append(yhat[0,0,:])
		# update state
		state = [h, c]
		# update target sequence
		target_seq = yhat
	return array(output)

# decode a one hot encoded string
def one_hot_decode(encoded_seq):
	return [argmax(vector) for vector in encoded_seq]

#converting to np arrays
X1 = np.squeeze(array(X1), axis=1) 
X2 = np.squeeze(array(X2), axis=1) 
y = np.squeeze(array(y), axis=1) 


X1_test = np.squeeze(array(X1_test), axis=1) 
X2_test = np.squeeze(array(X2_test), axis=1) 
y_test = np.squeeze(array(y_test), axis=1) 

# define model
train, infenc, infdec = define_models(n_features, no_out, 128)
train.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])

# train model
print(X1_test.shape,X2_test.shape,y_test.shape)
train.fit([X1, X2], y, epochs=80, batch_size = 64)

#predicting
correct = 0 #stores no of correct predictions
for x in xrange(0, len(X1_test)):
	print "#################################### new prediction :"
	sam = X1_test[x] #sample as input
	sam = sam[newaxis, :, :]
	target = predict_sequence(infenc, infdec, sam, max_out_len, no_out) #prediction
	#comparing with original result
	if array_equal(one_hot_decode(y_test[x]), one_hot_decode(target)):
		correct = correct+1
	slots = one_hot_decode(target) #contains slots as integer
	sen_slot = "" #contains slots as text
	for y in slots:
		if int_slot[y] == "#####":
			continue
		else:
			sen_slot = sen_slot + " " + int_slot[y]
	print "prediction:"
	print sen_slot #printing predicted slots
	print "Actual result:"
	print y_sen[x] #printing original slots
	print "Query:"
	print test_sen[x] #printing test sentece
	
print correct