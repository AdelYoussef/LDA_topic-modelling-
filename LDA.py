
from sklearn.datasets import fetch_20newsgroups
import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *
import numpy as np
import pandas as pd
np.random.seed(400)
import nltk



newsgroups_train = fetch_20newsgroups(subset='train', shuffle = True)
newsgroups_test = fetch_20newsgroups(subset='test', shuffle = True)

nltk.download('omw-1.4')
nltk.download('wordnet')
stemmer = SnowballStemmer("english")


class LDA :
        
        
    def init(self,train , test):
        self.data_train = train
        self.data_test = test
        
        nltk.download('omw-1.4')
        nltk.download('wordnet')
        self.stemmer = SnowballStemmer("english")
        
        return self.data_train, self.data_test
    
        
    def parsing(self, data):
        result=[]
        for token in gensim.utils.simple_preprocess(data) :
            if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
                
                Lemmatizer = WordNetLemmatizer().lemmatize(token, pos = 'v') 
                stemmer_Lemmatizer =  self.stemmer.stem(Lemmatizer)
                result.append(stemmer_Lemmatizer)
                
        return result
    
    
    def preprocess(self, data):
        
        processed_docs = []
    
        for doc in data:
            processed_docs.append(self.parsing(doc))
            
        return processed_docs 

    
    def doc2bow_dict(self,data):
        self.dictionary = gensim.corpora.Dictionary(data)
        self.bow_corpus = [self.dictionary.doc2bow(doc) for doc in data]
        return self.bow_corpus, self.dictionary
    

    def filter_docs(self,min_reps , min_occr , keep_most_freq = None):


        self.dictionary.filter_extremes(no_below = min_reps, no_above = min_occr , keep_n= keep_most_freq)
        
        
        
    def LDA_model(self,num_topics , dic, epochs, cores=2):
        
        if self.multicore == True:
            print("training begins ++++++ using multicore \n")
            self.lda_model =  gensim.models.LdaMulticore(self.bow_corpus, num_topics = num_topics, id2word = dic, passes = epochs, workers = cores)
            
        elif self.multicore == False:
            print("training begins ++++++ using singlecore \n")
            self.lda_model = gensim.models.LdaModel(self.bow_corpus, num_topics = num_topics, id2word = dic, passes = epochs)
            
        return self.lda_model
        
    def save_model(self,path):
      path = path + "LDA_model"
      self.lda_model.save(path)

    def train(self, path = "", save_model = True , Filter = True , multicore = True , cores = 2 , batch_size = 10 , epochs = 50 ,  num_topics=10 ):
        self.multicore = multicore
        self.Filter = Filter
        self.processed_docs = self.preprocess(self.data_train)
        
        self.doc2bow_dict(self.processed_docs)
        
        
        if self.Filter == True:
            self.filter_docs(min_reps = 15, min_occr = 0.1, keep_most_freq = None)
        
        
        self.LDA_model(num_topics = num_topics, dic = self.dictionary, epochs = epochs)
        print("training successful \n")

        if save_model == True:
            self.save_model(path)
        
    def classify(self,data):
        predict = []
        for doc in (data):
          processed = self.parsing(doc)
          bow_vector = self.dictionary.doc2bow(processed)
          vector =  np.array(self.lda_model[bow_vector])
          vec = np.max(vector, axis=0)
          predict.append(vec)
          
        return predict     
        
       
            
train = fetch_20newsgroups(subset='train', shuffle = True)
test = fetch_20newsgroups(subset='test', shuffle = True)
  
mylda = LDA()

mytrain , mytest = mylda.init(train.data , test.data)


mylda.train(Filter = False , multicore = False , batch_size = 10 , epochs = 50 ,num_topics=10)


unseen_document = []
unseen_document.append(newsgroups_test.data[100])
unseen_document.append(newsgroups_test.data[50])
unseen_document.append(newsgroups_test.data[60])

predictions = mylda.classify(unseen_document)

