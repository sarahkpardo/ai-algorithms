import assignment5submission as models
import numpy as np
import sys

if(sys.version_info[0] < 3):
    raise Exception("This assignment must be completed using Python 3")

def load_data(path):
    data = np.genfromtxt(path, delimiter=',', dtype=float)
    return data[:,:-1], data[:,-1].astype(int)

X, y = load_data("county_statistics.csv")

from sklearn import metrics
from sklearn.metrics import silhouette_score
import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering

# K-means
k = 3
t = 50 # max iterations
truncate = X.shape[0]
# sarah:
sarah_kmeansmodel = models.K_MEANS(k, t)
sarah_kmeanslabels = sarah_kmeansmodel.train(X[:truncate, :])
sarah_kmeansscore = silhouette_score(X[:truncate, :], sarah_kmeanslabels,
                                            metric='euclidean')
print('sarah kmeans score:', sarah_kmeansscore)
# Sklearn:
sklearn_kmeansmodel = KMeans(n_clusters=k, random_state=1).fit(X[:truncate, :])
sklearn_kmeanslabels = sklearn_kmeansmodel.labels_
sklearn_kmeansscore = silhouette_score(X[:truncate, :], sklearn_kmeanslabels,
                                                  metric='euclidean')
print('sklearn kmeans score:', sklearn_kmeansscore)

# AGNES
k = 8

# sarah:
sarah_agnesmodel = models.AGNES(k)
import time
start = time.time()
sarah_agneslabels = sarah_agnesmodel.train(X[:truncate, :])
end = time.time()
print("sarah agnes time:", end - start)
#print('agnes labels', sarah_agneslabels)
sarah_agnesscore = silhouette_score(X[:truncate, :], sarah_agneslabels,
                                            metric='euclidean')
print('sarah agnes score:', sarah_agnesscore)
# Sklearn
sklearn_agnesmodel = AgglomerativeClustering(n_clusters=k).fit(X[:truncate, :])
sklearn_agneslabels = sklearn_agnesmodel.labels_
sklearn_agnesscore = silhouette_score(X[:truncate, :], sklearn_agneslabels,
                                            metric='euclidean')
print('sklearn agnes score:', sklearn_agnesscore)
print(np.unique(sarah_agneslabels))