import imp
import matplotlib.pyplot as plt
from matplotlib import style
from sklearn.decomposition import PCA
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import adjusted_rand_score
from sklearn.feature_extraction.text import CountVectorizer
import nltk
import re
import os
from nltk.stem.snowball import SnowballStemmer
import pickle


style.use('ggplot')
colors = 10*["g","r","c","b","k"]

stopwords = nltk.corpus.stopwords.words('english')

stemmer = SnowballStemmer("english")

'''
Replace demo.txt by documents.text. Also, you can get ride of graph function in k7 file which plot dataset as we have gotten one graph for k5 version. 

'''
file = open("documents.txt", "rt")
contents = file.read()         
file.close()                   

data = contents.split("\n")

def tokenize_and_stem(text):
    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
    tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    stems = [stemmer.stem(t) for t in filtered_tokens]
    return stems

tfidf_vectorizer = TfidfVectorizer(max_df=0.8, max_features=200000,
                                 min_df=0.2, stop_words='english',
                                 use_idf=True, tokenizer=tokenize_and_stem, ngram_range=(1,3))

tfidf_matrix = tfidf_vectorizer.fit_transform(data).todense()

pca = PCA(n_components = 2).fit(tfidf_matrix)
data2D = pca.transform (tfidf_matrix)
plt.scatter(data2D[:,0], data2D[:,1])
plt.show()

pickle.dump(tfidf_matrix, open( "save.p", "wb" ) )



class K_Means:
    def __init__(self, k=10, tol=0.001, max_iter=100):
        self.k = k
        self.tol = tol
        self.max_iter = max_iter

    def fit(self,data):

        self.centroids = {}

        for i in range(self.k):
            self.centroids[i] = data[i]

        for i in range(self.max_iter):
            self.classifications = {}

            for i in range(self.k):
                self.classifications[i] = []

            for featureset in data:
                distances = [np.linalg.norm(featureset-self.centroids[centroid]) for centroid in self.centroids]
                classification = distances.index(min(distances))
                self.classifications[classification].append(featureset)

            prev_centroids = dict(self.centroids)

            for classification in self.classifications:
                self.centroids[classification] = np.average(self.classifications[classification],axis=0)

            optimized = True

            for c in self.centroids:
                original_centroid = prev_centroids[c]
                current_centroid = self.centroids[c]
                if np.sum((current_centroid-original_centroid)/original_centroid*100.0) > self.tol:
                    #print(np.sum((current_centroid-original_centroid)/original_centroid*100.0))
                    optimized = False

            if optimized:
                break

clf = K_Means()
clf.fit(tfidf_matrix)

print ("printing centroid values:")

for centroid, value in (clf.centroids).items():
    print (centroid, "value is", value)

print("calculating SSE")

sum_of_clusters = 0
sum_of_points = 0

for classification in clf.classifications:
    m = clf.centroids[classification]
    for featureset in clf.classifications[classification]:
        sum_of_points = sum_of_points + ((np.linalg.norm(featureset - m))*(np.linalg.norm(featureset - m)))
    sum_of_clusters = sum_of_clusters + sum_of_points
    sum_of_points = 0
        
print ("SSE for kmeans with k = 10")
print (sum_of_clusters)
