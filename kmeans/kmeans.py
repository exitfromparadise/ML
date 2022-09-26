"""
following https://realpython.com/k-means-clustering-python/

k-means clustering: non-deterministic, unsupervised machine learning algorithm for partitional clustering

for spherical or well shaped clusters, elbow and silhouette coefficients can be used to quantifiy clustering performance

"""

import matplotlib.pyplot as plt
from kneed import KneeLocator
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import sklearn

def create_blob_data():
    # generate data
    features, true_labels = make_blobs(
        n_samples = 200,
        centers = 3,
        cluster_std = 2.75,
        random_state = 42,
    )

    # feature scaling: transform numerical values to same scale especially for distance-based algorithms
    scaler = sklearn.preprocessing.StandardScaler()
    scaled_features = scaler.fit_transform(features)
    return scaled_features, true_labels

def simple_kMeans():
    print("running simple kmeans")

    scaled_features = create_blob_data()

    # clustering with KMeans estimator class from scikit-learn
    kmeans_kwargs = {
        "init" : "random",
        "n_clusters" : 3,
        "n_init" : 10,
        "max_iter" : 300,
        "random_state" : 42,
    }

    kmeans = KMeans(**kmeans_kwargs)
    kmeans.fit(scaled_features)

    print("lowest SSE value: ", kmeans.inertia_)
    #print("location of the centroids: ", kmeans.cluster_centers_)
    print("number of iterations for convergence: ", kmeans.n_iter_)
    #print("labels: ", kmeans.labels_)


def advanced_kMeans():
    """
    study optimised number of clusters using the elbow method and silhouette coefficient which are not based on ground truth labels
    """
    print("running advanced kmeans")

    elbow_method = 0 # else use silhouette coefficient

    scaled_features = create_blob_data()

    kmeans_kwargs = {
        "init" : "random",
        "n_init" : 10,
        "max_iter" : 300,
        "random_state" : 42,
    }

    if elbow_method:
        sse = []
        for k in range(1,11):
            kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
            kmeans.fit(scaled_features)
            sse.append(kmeans.inertia_)

        # plot error (sse) vs. umber of clusters
        # sse = sum of the squared distance between centroid and each member of the cluster
        #plt.style.use("fivethirtyeight")
        plt.figure(0)
        plt.plot(range(1, 11), sse)
        plt.xticks(range(1, 11))
        plt.xlabel("Number of Clusters")
        plt.ylabel("SSE")
        plt.title("elbow point - best trafe of between error and number of clusters")
        plt.tight_layout()
        plt.savefig("sse_vs_nclusters.png")

        # determine elbow point using kneed programmatically
        kl = KneeLocator(
            range(1, 11),
            sse,
            curve = "convex",
            direction = "decreasing",
        )
        print("elbow point:", kl.elbow)

    else:

        # silhouette coefficient
        # quanitifies in the range -1 to 1 how well data point fits into assigned cluster based on, larger number means closer to their cluster than to others
        # how close the data points are to other points in cluster and how far away points from other clusters are

        silhouette_coefficients = []

        for k in range(2,11):
            kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
            kmeans.fit(scaled_features)
            score = silhouette_score(scaled_features, kmeans.labels_)
            silhouette_coefficients.append(score)

        print(silhouette_coefficients)
        plt.figure(1)
        plt.plot(range(2, 11), silhouette_coefficients)
        plt.xticks(range(2, 11))
        plt.xlabel("Number of Clusters")
        plt.ylabel("Silhouette Coefficient")
        plt.tight_layout()
        plt.savefig("silhouette_coeff_vs_nclusters.png")

if __name__ == "__main__":
    simple_kMeans()
    advanced_kMeans()
