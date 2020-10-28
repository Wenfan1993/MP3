# *- coding: utf-8 -*-
"""
Created on Sun Oct 25 22:07:08 2020

@author: Wenxi
"""
import numpy as np
import os
from collections import Counter
from plsa import Corpus

path = r'C:\Users\Wenxi\Desktop\git_projects\MP3'
os.chdir(path)

documents_path = 'data/test.txt'
likelihoods = []

def normalize(input_matrix):
    """
    Normalizes the rows of a 2d input_matrix so they sum to 1
    """

    row_sums = input_matrix.sum(axis=1)
    try:
        assert (np.count_nonzero(row_sums)==np.shape(row_sums)[0]) # no row should sum to zero
    except Exception:
        raise Exception("Error while normalizing. Row(s) sum to zero")
    new_matrix = input_matrix / row_sums[:, np.newaxis]
    return new_matrix


##===function 1
##===to get - 1. documents = []    2.number_of_documents=int
with open(documents_path, 'r') as file:
    text = file.readlines()

#targets 1.1
documents = []

for txt in text:
    txt_doc = []
    for item in txt.split(' '):
        if item[0]==str(0) or item[0]==str(1):
            item = item[1:]
        
        item = item.replace(' ','').replace('\t','').replace('\n','').lower()
        if len(item)>0:
            txt_doc.append(item)    
        
    documents.append(txt_doc)
#target 1.2
number_of_documents = len(documents)


##===function 2
##===to get - 1. vocabulary = []    2.vocabulary_size=int
vocabulary = []

for txt in documents:
    for word in txt:
        if word not in vocabulary:
            vocabulary.append(word)

vocabulary = list(set(vocabulary))        
vocabulary_size = len(vocabulary)     


#prepare other 
number_of_topics = 2
topic_prob = np.zeros([number_of_documents, number_of_topics, vocabulary_size], dtype=np.float)

##===function 3
##===to get - 1. term_doc_matrix = np.array
   
matrix = []

for txt in documents:
    cnt = Counter()
    for word in txt:
        cnt[word]+=1

    def return_key(key):
        try:
            return cnt[key]
        except KeyError:
            return 0

    matrix.append(list(map(return_key, vocabulary)))

term_doc_matrix = np.array(matrix)

##===function 4
##===document_topic_prob: P(z | d):
##===topic_word_prob: and P(w | z)
##===to get - 1. term_doc_matrix = np.array


document_topic_prob = np.random.rand(number_of_documents,number_of_topics)
document_topic_prob = normalize(document_topic_prob)
        
topic_word_prob = np.random.rand(number_of_topics,vocabulary_size)
topic_word_prob = normalize(topic_word_prob)


##===function 5
##===document_word_topic_prob: P(z | d, w)
document_word_prob = document_topic_prob.dot(topic_word_prob)#size d*w = 

for d in range(number_of_documents):
    for w in range(vocabulary_size):
        for z in range(number_of_topics):
            topic_prob[d,z,w] = (document_topic_prob[d].reshape(-1)*topic_word_prob[:,w].reshape(-1))[z]/(document_word_prob[d,w])

##===function 6
##===document_topic_prob: P(z | d)

term_doc_matrix_reshape = term_doc_matrix.reshape(term_doc_matrix.shape[0],1,term_doc_matrix.shape[1])
term_times_document_word_topic = term_doc_matrix_reshape * topic_prob

document_word_prob_ = term_times_document_word_topic.sum(axis=2)/(term_times_document_word_topic.sum(axis=2)).sum(axis=1).reshape(-1,1)
document_word_prob = normalize(document_word_prob_)

topic_word_prob_ = term_times_document_word_topic.sum(axis=0)/(term_times_document_word_topic.sum(axis=0)).sum(axis=1).reshape(-1,1)
topic_word_prob = normalize(topic_word_prob_)


likelihood = np.sum(term_doc_matrix * np.log(document_topic_prob.dot(topic_word_prob)))

likelihoods.append(likelihood)





documents_path = 'data/test.txt'
corpus = Corpus(documents_path)  # instantiate corpus
corpus.build_corpus()
corpus.build_vocabulary()
print(corpus.vocabulary)
print("Vocabulary size:" + str(len(corpus.vocabulary)))
print("Number of documents:" + str(len(corpus.documents)))
number_of_topics = 2
max_iterations = 50
epsilon = 0.001
corpus.plsa(number_of_topics, max_iterations, epsilon)

corpus.document_topic_prob
