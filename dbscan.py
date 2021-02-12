import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib import style
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import adjusted_rand_score
from sklearn.feature_extraction.text import CountVectorizer
import nltk
import re
import os
from sklearn.neighbors import KNeighborsClassifier  
from sklearn.neighbors import NearestNeighbors
import pickle
from nltk.stem.snowball import SnowballStemmer

style.use('ggplot')

colors = 10*["g","r","c","b","k"]

'''
stopwords = nltk.corpus.stopwords.words('english')

stemmer = SnowballStemmer("english")

# Change file to documents.txt

a = input()
file = open(a, "rt")
contents = file.read()         
file.close()                   

contents = unicode(contents, errors = 'ignore')
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

'''
tfidf_matrix = pickle.load(open("save.p", "rb"))

#ask about elbow graph to professor. and eps value.
def eps_value_calculation(cluster_value):
    ns = cluster_value
    nbrs = NearestNeighbors(n_neighbors=ns).fit(tfidf_matrix)
    distances, indices = nbrs.kneighbors(tfidf_matrix)
    distanceDec = sorted(distances[:,ns-1], reverse = False)
    plt.plot(indices[:,0], distanceDec)
    plt.show() 
'''
UNCLASSIFIED = False
NOISE = None

def dist(p,q):
	return math.sqrt(np.power(p-q,2).sum())

def eps_neighborhood(p,q,eps):
	return dist(p,q) < eps

def region_query(m, point_id, eps):
    n_points = m.shape[1]
    seeds = []
    for i in range(0, n_points):
        if eps_neighborhood(m[:,point_id], m[:,i], eps):
            seeds.append(i)
    return seeds

def expand_cluster(m, classifications, point_id, cluster_id, eps, min_points):
    seeds = region_query(m, point_id, eps)
    if len(seeds) < min_points:
        classifications[point_id] = NOISE
        return False
    else:
        classifications[point_id] = cluster_id
        for seed_id in seeds:
            classifications[seed_id] = cluster_id
            
        while len(seeds) > 0:
            current_point = seeds[0]
            results = region_query(m, current_point, eps)
            if len(results) >= min_points:
                for i in range(0, len(results)):
                    result_point = results[i]
                    if classifications[result_point] == UNCLASSIFIED or \
                       classifications[result_point] == NOISE:
                        if classifications[result_point] == UNCLASSIFIED:
                            seeds.append(result_point)
                        classifications[result_point] = cluster_id
            seeds = seeds[1:]
        return True
        
def dbscan(m, eps, min_points):
    cluster_id = 1
    n_points = m.shape[1]
    classifications = [UNCLASSIFIED] * n_points
    for point_id in range(0, n_points):
        point = m[:,point_id]
        if classifications[point_id] == UNCLASSIFIED:
            if expand_cluster(m, classifications, point_id, cluster_id, eps, min_points):
                cluster_id = cluster_id + 1
    return classifications

# ask about SSE calculation? coorect or not? 
def calculate_SSE(dic_of_cluster):
    sum_of_points =0
    sum_of_cluster = 0

    for classification in dic_of_cluster:
        val = dic_of_cluster.get(classification)
        centroid = np.mean(val, axis=0)
        for featureset in dic_of_cluster[classification]:
            sum_of_points = sum_of_points + ((np.linalg.norm(featureset-centroid))*(np.linalg.norm(featureset-centroid)))
        sum_of_cluster = sum_of_cluster + sum_of_points
        sum_of_points = 0

    return sum_of_cluster

def making_dir_of_cluster(classification_array, m):
    modified_classification_array = [x for x in classification_array if x is not None]

    number_of_clusters = max(modified_classification_array)
    print ("number_of_clusters")
    print(number_of_clusters)

    dic_of_cluster = {}

    for i in range(number_of_clusters):
        dic_of_cluster[i] = []

    for i in range(len(classification_array)):
        if classification_array[i] == None:
            continue
        element = m[i]
        idx = classification_array[i]
        corrected_idx = idx-1
        dic_of_cluster[corrected_idx].append(element)
    
    SSE_of_cluster = calculate_SSE(dic_of_cluster)
    print("SSE_of_dbscan")
    print(SSE_of_cluster)

def test_dbscan():
    m = tfidf_matrix
    m_trans = np.transpose(m)
    dpcount = m.shape[0]
    print ("number of rows i.e. number of data points.", dpcount)
    print ("min_points is natural log of number of data points.")
    nl = math.log(dpcount)
    print ("natural log value is", nl)
    print("enter value close to natural log value for min_points")
    min_points = int(input())
    eps_value_calculation(min_points)
    print ("calculate eps value from graph shown previously.")
    print ("enter eps value")
    eps = float(input())
    classification_array = dbscan(m_trans, eps, min_points)
    making_dir_of_cluster(classification_array, m)

test_dbscan()
# ask about minpoint value

'''
min_points= 11
eps_value_calculation(min_points)