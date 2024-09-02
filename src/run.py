import pandas as pd
import re
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV

def preprocess(df):

  print('preprocessing')
  df['text']=df.textcontent.apply(lambda x :re.sub(r'[^\w\s]',' ',str(x)))
  df.text=df.text.apply(lambda x : re.sub(r'[^\u0600-\u06FF\s]+',' ',str(x)))
  df.text=df.text.apply(lambda x : re.sub(r'[\u06F0-\u06F9]',' ',str(x)))
  df.text=df.text.apply(lambda x : re.sub(r'[\u0660-\u0669]',' ',str(x)))
  df.text=df.text.apply(lambda x : re.sub(r'\s{2,}',' ',str(x)))
  df.text=df.text.apply(lambda x : re.sub(r'[\u200c]',' ',str(x)))

  return df

class Word2VecVectorizer:
  def __init__(self, model,embedding_size):
    print("Loading in word vectors...")
    self.word_vectors = model
    self.embedding_size=embedding_size
    print("Finished loading in word vectors")

  def fit(self, data):
    pass

  def transform(self, data):
    X = np.zeros((len(data), self.embedding_size))
    n = 0
    emptycount = 0
    for sentence in data:
      tokens = sentence.split(' ')
      vecs = []
      m = 0
      for word in tokens:
        try:
          vec = self.word_vectors[word]
          vecs.append(vec)
          m += 1
        except KeyError:
          pass
      if len(vecs) > 0:
        vecs = np.array(vecs)
        X[n] = vecs.mean(axis=0)
      else:
        emptycount += 1
      n += 1
    print("Numer of samples with no words found: %s / %s" % (emptycount, len(data)))
    return X

  def fit_transform(self, data):
    self.fit(data)
    return self.transform(data)

def test_big_data(data):

  with open('KNN_model.pkl','rb') as f_KNN:
    KNN=pickle.load(f_KNN)

  print("shape",data.shape)
  data=preprocess(data)
  from gensim.models.keyedvectors import KeyedVectors
  file_path = 'insta_wchr.vec'
  model = KeyedVectors.load_word2vec_format(file_path)
  vectorizer = Word2VecVectorizer(model,100)
  X = vectorizer.fit_transform(data.text.values)

  data['pred_KNN']=KNN.predict(X)

  data[['pred_KNN','textcontent']].to_excel('label_test.xlsx')

  print('predict done')


data=pd.read_excel("test.xlsx",engine="openpyxl")
test_big_data(data)

