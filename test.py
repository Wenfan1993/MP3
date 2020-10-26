# *- coding: utf-8 -*-
"""
Created on Sun Oct 25 22:07:08 2020

@author: Wenxi
"""
import numpy as np
import os

path = r'C:\Users\Wenxi\Desktop\git_projects\MP3'
os.chdir(path)

from plsa import Corpus
documents_path = 'data/test.txt'
corpus = Corpus(documents_path)  # instantiate corpus
corpus.build_corpus()
corpus.build_vocabulary()
print(corpus.vocabulary)
print("Vocabulary size:" + str(len(corpus.vocabulary)))
print("Number of documents:" + str(len(corpus.documents)))
number_of_topics = 2
max_iterations = 50
max_iter = 50
epsilon = 0.001
corpus.plsa(number_of_topics, max_iterations, epsilon)

        corpus.build_term_doc_matrix()
        
        a = corpus.term_doc_matrix
        b = corpus.documents
        
        # Create the counter arrays.
        
        # P(z | d, w)
        corpus.topic_prob = np.zeros([corpus.number_of_documents, number_of_topics, corpus.vocabulary_size], dtype=np.float)

        # P(z | d) P(w | z)
        corpus.initialize(number_of_topics, random=True)

        # Run the EM algorithm
        current_likelihood = 0.0

        for iteration in range(max_iter):
            print("Iteration #" + str(iteration + 1) + "...")
            #iteration = 0
            
            corpus.expectation_step()
            
            corpus.maximization_step(number_of_topics)
            
            if abs(corpus.calculate_likelihood(number_of_topics))<epsilon:
                break

        corpus.w_d = corpus.topic_word_prob.T.dot(corpus.document_topic_prob.T)#size w*d
        
        d_topic_prob = []
        
        for d in range(corpus.number_of_documents):
            w_topic_prob = []
            for w in range(len(corpus.vocabulary)):
                w_topic_prob.append([(corpus.document_topic_prob[d,k]*corpus.topic_word_prob[k,w])/corpus.w_d[w,d] for k in range(corpus.number_of_topics)])
            d_topic_prob.append(w_topic_prob)
        
        corpus.topic_prob = np.array(d_topic_prob)
