#QUORA COMPETITION
import sys  
reload(sys)  
sys.setdefaultencoding('utf8')
import StringIO
import csv
import re, math
from collections import Counter
from nltk.corpus import stopwords
from nltk import word_tokenize
from time import time
import logging
import string
#from nltk import download
from sklearn import svm
import os
import codecs
import pandas as pd
import numpy as np
from gensim.models import KeyedVectors

"""
Requirements:

* nltk stopswords -> http://www.nltk.org/data.html ()
* gensim library: https://radimrehurek.com/gensim/install.html (pip install --upgrade gensim)
* Word2Vec Google News Vectors: https://code.google.com/archive/p/word2vec/ or https://github.com/mmihaltz/word2vec-GoogleNews-vectors

"""


def load_google_news_vectors(google_news_vectors_path):

	start = time()

	if not os.path.exists(google_news_vectors_path):
		raise ValueError("Error. You need to download the google news model: https://code.google.com/archive/p/word2vec/")

	model = KeyedVectors.load_word2vec_format(google_news_vectors_path, binary=True)
	print('Loading word2vec Google News Vectors took %.2f seconds to run.' % (time() - start))
	return model

def get_cosine(vec1, vec2):
	"""
	"""

	intersection = set(vec1.keys()) & set(vec2.keys())
	numerator = sum([vec1[x] * vec2[x] for x in intersection])

	sum1 = sum([vec1[x]**2 for x in vec1.keys()])
	sum2 = sum([vec2[x]**2 for x in vec2.keys()])
	denominator = math.sqrt(sum1) * math.sqrt(sum2)

	if not denominator:
		return 0.0
	else:
		return float(numerator) / denominator

def text_to_vector(text):
	"""
	"""

	WORD = re.compile(r'\w+')
	words = WORD.findall(text)
	return Counter(words)


def remove_stopwords(text):

	# To circumvent problems with lower function on non string obejcts
	text_as_string = str(text)

	# split the text into tokens
	#tokens = text.strip().lower().split()

	# word_tokenize seems to be the better choice: https://stackoverflow.com/questions/17390326/getting-rid-of-stop-words-and-document-tokenization-using-nltk
	#print(text)
	tokens = word_tokenize(text_as_string.lower())

	[x.encode('utf-8') for x in tokens]

	# remove stopwords and punctuations like questionmarks
	filtered_words = [word for word in tokens if word not in stopwords.words('english') + list(string.punctuation)]
	return filtered_words

def check_for_nan_values(dataframe):
	pass


def generate_dataframe_without_stopwords(input_dataframe):

	# Create new columns in the dataframe
	input_dataframe["q1_no_stop"] = ""
	input_dataframe["q2_no_stop"] = ""

	print("Removing stopwords...")

	# Convert the columns to lists for better performance
	question1_list = input_dataframe["question1"].tolist()
	question2_list = input_dataframe["question2"].tolist()

	# Remove stopswords and punctuation
	question1_no_stop = list(map(remove_stopwords, question1_list))
	question2_no_stop = list(map(remove_stopwords, question2_list))

	# Convert the lists without stopwords to series so you can easily insert them into the dataframe
	se1 = pd.Series(question1_no_stop)
	se2 = pd.Series(question2_no_stop)

	# Put the newly created questions without stopswords into the dataframe
	input_dataframe["q1_no_stop"] = se1.values
	input_dataframe["q2_no_stop"] = se2.values

	print(input_dataframe.head())
	return input_dataframe



def main():

	CURRENT_WORKING_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
	print("Current working directory:" + CURRENT_WORKING_DIRECTORY)
	GOOGLE_NEWS_VECTORS_PATH = os.path.join(CURRENT_WORKING_DIRECTORY, 'GoogleNews-vectors-negative300.bin.gz')
	TRAINING_SET_PATH = os.path.join(CURRENT_WORKING_DIRECTORY, 'input', 'train.csv')
	TEST_TEST_PATH = os.path.join(CURRENT_WORKING_DIRECTORY, 'input', 'test.csv')

	print("Loading datasets...");

	TRAIN_DATAFRAME = pd.read_csv(TRAINING_SET_PATH, encoding = 'utf8', keep_default_na=False, na_values=['nan',''])
	TEST_DATAFRAME = pd.read_csv(TEST_TEST_PATH, encoding = 'utf8', keep_default_na=False, na_values=['nan',''])


	print("Computing missing values for Training set...")
	print(np.where(pd.isnull(TRAIN_DATAFRAME)))
	print(TRAIN_DATAFRAME.isnull().values.sum())

	print("Computing missing values for Test set...")
	print(np.where(pd.isnull(TEST_DATAFRAME)))
	print(TEST_DATAFRAME.isnull().values.sum())

	#exit()

	train_no_stop = generate_dataframe_without_stopwords(TRAIN_DATAFRAME)
	test_no_stop = generate_dataframe_without_stopwords(TEST_DATAFRAME)


	print(train_no_stop.head())
	print(test_no_stop.head())

	train_no_stop.to_csv("train_without_stopwords.csv")
	test_no_stop.to_csv("test_without_stopwords.csv")


	# print(TRAIN_DATAFRAME.head())
	# print(TEST_DATAFRAME.head())

	# TRAIN_DATAFRAME.to_csv("train_without_stopwords.csv")
	# TEST_DATAFRAME.to_csv("test_without_stopwords.csv")


	#download('stopwords')  # Download stopwords list.
	logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s')



	exit()

	model = load_google_news_vectors(GOOGLE_NEWS_VECTORS_PATH)


	X = []
	y = []

	with codecs.open('output.txt', 'w', 'utf-8') as outputfile:
		outputfile.write('\t'.join(['id','qid1','qid2','q1','q2','dup','q1_no_stop','q2_no_stop','c','cf','dis']) + '\n')

		#import training data
		with open('output_train_no_stop', 'r') as myfile:
				quora = myfile.readlines()
				header = quora[0]
				#id, qid1, qid2, question1, question2, is_duplicate

				#for item in quora[1:len(quora)]:
				for item in quora[1:len(quora)]:
					print item
					item_list = item.split('\t')
					#print item_list

					line_id = item_list[0]
					qid1 = item_list[1]
					qid2 = item_list[2]
					question1 = item_list[3]
					question2 = item_list[4]
					is_duplicate = item_list[5]
					question1_no_stop = item_list[6]
					question2_no_stop = item_list[7]

					#Cosine similarity
					vector1 = text_to_vector(question1)
					vector2 = text_to_vector(question2)

					cosine = get_cosine(vector1, vector2)
					#potentially not good because some questions get high cosine similarity even though they lack a specific term

					#Cosine stopwords excluded
					question1_list = question1.split()
					question2_list = question2.split()

					vectorF1 = text_to_vector(q1_no_stop)
					vectorF2 = text_to_vector(q2_no_stop)

					cosineF = get_cosine(vectorF1,vectorF2)

					#Word Mover's distance (using Word2Vec embeddings)
					# The distance between two text documents A and B is the minimum cumulative
					# distance that words from document A need to travel to match exactly the point cloud of document

					model.init_sims(replace=True)

					distance = model.wmdistance(q1_no_stop,q2_no_stop)
					#print 'distance = %.4f' % distance

					new_item_list = [cosine, cosineF, distance]

					X.append(new_item_list)
					y.append(is_duplicate)

					#Wordnet synsets
					#Jaccard index
					#Euclidean distance
					#Pearson's R

					outputfile.write(u'\t'.join(map(unicode,[
					line_id,qid1,qid2,question1,question2,is_duplicate,q1_no_stop,q2_no_stop,cosine,cosineF,distance])) + '\n')


	print X
	print y

	clf = svm.SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
	    decision_function_shape=None, degree=3, gamma='auto', kernel='rbf',
	    max_iter=-1, probability=False, random_state=None, shrinking=True,
	    tol=0.001, verbose=False)
	clf.fit(X, y)

	output_test = []

	with codecs.open('test_output.txt', 'w', 'utf-8') as outputfile:
		outputfile.write('\t'.join(['test_id','is_duplicate']) + '\n')

		with open('output_test_no_stop.txt', 'r') as testfile:
					quora = testfile.readlines()
					header = quora[0]
					#id, qid1, qid2, question1, question2, is_duplicate

					#for item in quora[1:len(quora)]:
					for item in quora[1:len(quora)]:
						item_list = item.split('\t')
						#print item_list

						line_id = item_list[0]
						question1 = item_list[1]
						question2 = item_list[2]
						q1_no_stop = item_list[3]
						q2_no_stop = item_list[4]

						#Cosine similarity
						vector1 = text_to_vector(question1)
						vector2 = text_to_vector(question2)

						cosine = get_cosine(vector1, vector2)
						#potentially not good because some questions get high cosine similarity even though they lack a specific term

						#Cosine stopwords excluded
						vectorF1 = text_to_vector(q1_no_stop)
						vectorF2 = text_to_vector(q2_no_stop)

						cosineF = get_cosine(vectorF1,vectorF2)

						#Word Mover's distance (using Word2Vec embeddings)
						# The distance between two text documents A and B is the minimum cumulative
						# distance that words from document A need to travel to match exactly the point cloud of document B

						model.init_sims(replace=True)

						distance = model.wmdistance(q1_no_stop, q2_no_stop)
						#print 'distance = %.4f' % distance

						predict_list = [[cosine,cosineF,distance]]

						prediction = clf.predict(predict_list)

						outputfile.write(u'\t'.join(map(unicode,[line_id,prediction[0]])) + '\n')








if __name__ == "__main__":
    main()