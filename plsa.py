import numpy as np
import math
from collections import Counter


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

       
class Corpus(object):

    """
    A collection of documents.
    """

    def __init__(self, documents_path):
        """
        Initialize empty document list.
        """
        self.documents = []
        self.vocabulary = []
        self.likelihoods = []
        self.documents_path = documents_path
        self.term_doc_matrix = None 
        self.document_topic_prob = None  # P(z | d)
        self.topic_word_prob = None  # P(w | z)
        self.topic_prob = None  # P(z | d, w)

        self.number_of_documents = 0
        self.vocabulary_size = 0

#function 1
    def build_corpus(self):
        """
        Read document, fill in self.documents, a list of list of word
        self.documents = [["the", "day", "is", "nice", "the", ...], [], []...]
        
        Update self.number_of_documents
        """
        with open(self.documents_path, 'r') as file:
            text = file.readlines()
        
        for txt in text:
            txt_doc = []
            for item in txt.split(' '):
                if item[0]==str(0) or item[0]==str(1):
                    item = item[1:]
                
                item = item.replace(' ','').replace('\t','').replace('\n','').lower()
                if len(item)>0:
                    txt_doc.append(item)        
                
            self.documents.append(txt_doc)

        self.number_of_documents = len(self.documents)
            
#function2
    def build_vocabulary(self):
        """
        Construct a list of unique words in the whole corpus. Put it in self.vocabulary
        for example: ["rain", "the", ...]

        Update self.vocabulary_size
        """
        for txt in self.documents:
            for word in txt:
                if word not in self.vocabulary:
                    self.vocabulary.append(word)
        
        self.vocabulary = list(set(self.vocabulary))        
        self.vocabulary_size = len(self.vocabulary)        

#function 3
    def build_term_doc_matrix(self):
        """
        Construct the term-document matrix where each row represents a document, 
        and each column represents a vocabulary term.

        self.term_doc_matrix[i][j] is the count of term j in document i
        """

        matrix = []
        
        for txt in self.documents:
            cnt = Counter()
            for word in txt:
                cnt[word]+=1

            def return_key(key):
                try:
                    return cnt[key]
                except KeyError:
                    return 0

            matrix.append(list(map(return_key, self.vocabulary)))
        
        self.term_doc_matrix = np.array(matrix)

#function 4
    def initialize_randomly(self, number_of_topics):
        """
        Randomly initialize the matrices: document_topic_prob and topic_word_prob
        which hold the probability distributions for P(z | d) and P(w | z): self.document_topic_prob, and self.topic_word_prob

        Don't forget to normalize! 
        HINT: you will find numpy's random matrix useful [https://docs.scipy.org/doc/numpy-1.15.0/reference/generated/numpy.random.random.html]
        """
        self.number_of_topics = number_of_topics
        self.document_topic_prob = np.random.rand(self.number_of_documents,number_of_topics)
        self.document_topic_prob = normalize(self.document_topic_prob)
                
        self.topic_word_prob = np.random.rand(number_of_topics, self.vocabulary_size)
        self.topic_word_prob = normalize(self.topic_word_prob)
        
                                                                                               
    def initialize_uniformly(self, number_of_topics):
        """
        Initializes the matrices: self.document_topic_prob and self.topic_word_prob with a uniform 
        probability distribution. This is used for testing purposes.

        DO NOT CHANGE THIS FUNCTION
        """
        self.document_topic_prob = np.ones((self.number_of_documents, number_of_topics))
        self.document_topic_prob = normalize(self.document_topic_prob)

        self.topic_word_prob = np.ones((number_of_topics, len(self.vocabulary)))
        self.topic_word_prob = normalize(self.topic_word_prob)

    def initialize(self, number_of_topics, random=False):
        """ Call the functions to initialize the matrices document_topic_prob and topic_word_prob
        """
        print("Initializing...")

        if random:
            self.initialize_randomly(number_of_topics)
        else:
            self.initialize_uniformly(number_of_topics)

    def expectation_step(self):
        """ The E-step updates P(z | w, d)
        """
        print("E step:")
        
        self.document_word_prob = self.document_topic_prob.dot(self.topic_word_prob)#size d*w = 
        
        for d in range(self.number_of_documents):
            for w in range(self.vocabulary_size):
                for z in range(self.number_of_topics):
                    self.topic_prob[d,z,w] = (self.document_topic_prob[d].reshape(-1)*self.topic_word_prob[:,w].reshape(-1))[z]/(self.document_word_prob[d,w])

    def maximization_step(self, number_of_topics):
        """ The M-step updates P(w | z)
        """
        print("M step:")
        
        term_doc_matrix_reshape = self.term_doc_matrix.reshape(self.term_doc_matrix.shape[0],1,self.term_doc_matrix.shape[1])
        term_times_document_word_topic = term_doc_matrix_reshape * self.topic_prob

        document_word_prob_ = term_times_document_word_topic.sum(axis=2)/(term_times_document_word_topic.sum(axis=2)).sum(axis=1).reshape(-1,1)
        self.document_word_prob = normalize(document_word_prob_)

        topic_word_prob_ = term_times_document_word_topic.sum(axis=0)/(term_times_document_word_topic.sum(axis=0)).sum(axis=1).reshape(-1,1)
        self.topic_word_prob = normalize(topic_word_prob_)
    

    def calculate_likelihood(self, number_of_topics):
        """ Calculate the current log-likelihood of the model using
        the model's updated probability matrices
        
        Append the calculated log-likelihood to self.likelihoods

        """
        likelihood = np.sum(self.term_doc_matrix * np.log(self.document_topic_prob.dot(self.topic_word_prob)))
        self.likelihoods.append(likelihood)
        
        return likelihood

    def plsa(self, number_of_topics, max_iter, epsilon):

        """
        Model topics.
        """
        print ("EM iteration begins...")
        
        # build term-doc matrix
        self.build_term_doc_matrix()
        
        # Create the counter arrays.
        
        # P(z | d, w)
        self.topic_prob = np.zeros([self.number_of_documents, number_of_topics, self.vocabulary_size], dtype=np.float)

        # P(z | d) P(w | z)
        self.initialize(number_of_topics, random=True)

        # Run the EM algorithm
        current_likelihood = 0.0

        for iteration in range(max_iter):
            print("Iteration #" + str(iteration + 1) + "...")
            
            self.expectation_step()
            
            self.maximization_step(number_of_topics)
            
            loss = self.calculate_likelihood(number_of_topics)
            
            try:
                delta = abs(self.likelihoods[-1] - self.likelihoods[-2])
                if delta<epsilon:
                    break

            except IndexError:
                pass
            

def main():
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



if __name__ == '__main__':
    main()
