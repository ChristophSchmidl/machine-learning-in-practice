#QUORA COMPETITION
import sys  
reload(sys)  
sys.setdefaultencoding('utf8')
import StringIO
import csv
import re, math
from collections import Counter
#from nltk.corpus import stopwords
from time import time
import logging
#from nltk import download
from sklearn import svm
import os
import codecs
print 'hello'
start = time()
from gensim.models import KeyedVectors
print 'hello2'
current_folder = os.path.dirname(os.path.abspath(__file__))
google_news_vectors =  path = os.path.join(current_folder, 'GoogleNews-vectors-negative300.bin.gz')
print current_folder
print google_news_vectors

if not os.path.exists(google_news_vectors):
    raise ValueError("SKIP: You need to download the google news model")
print 'hello3'
model = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin.gz', binary=True)
print 'hello4'
print('Cell took %.2f seconds to run.' % (time() - start))
download('stopwords')  # Download stopwords list.
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s')

#define functions
WORD = re.compile(r'\w+')

def get_cosine(vec1, vec2):
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
     words = WORD.findall(text)
     return Counter(words)

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

