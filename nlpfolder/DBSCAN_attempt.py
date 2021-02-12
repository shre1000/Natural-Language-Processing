import numpy as np

from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import StandardScaler

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
from nltk.stem.snowball import SnowballStemmer
import pickle

tfidf_matrix = pickle.load(open("save.p", "rb"))

db = DBSCAN(eps=1, min_samples=11).fit(tfidf_matrix)

labels = db.labels_
print(labels)

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
#n_noise_ = list(labels).count(-1)

print('Estimated number of clusters: %d' % n_clusters_)
#print("V-measure: %0.3f" % metrics.v_measure_score(true, labels))

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

def making_dir_of_cluster(labels, m):
    modified_classification_array = [x for x in labels if x is not -1]

    number_of_clusters = max(modified_classification_array)
    print ("number_of_clusters")
    print(number_of_clusters)

    dic_of_cluster = {}

    for i in range(number_of_clusters):
        dic_of_cluster[i] = []

    for i in range(len(labels)):
        if labels[i] == -1:
            continue
        element = tfidf_matrix[i]
        idx = labels[i]
        corrected_idx = idx-1
        dic_of_cluster[corrected_idx].append(element)
    
    SSE_of_cluster = calculate_SSE(dic_of_cluster)
    print("SSE_of_dbscan")
    print(SSE_of_cluster)

