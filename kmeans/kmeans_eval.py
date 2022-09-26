from sklearn.cluster import DBSCAN
from sklearn.datasets import make_moons
from sklearn.metrics import adjusted_rand_score
import matplotlib.pyplot as plt
from kneed import KneeLocator
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import sklearn

def create_moon_data():
    features, true_labels = make_moons(
        n_samples = 250,
        noise = 0.05,
        random_state = 42
    )
    scaler = sklearn.preprocessing.StandardScaler()
    scaled_features = scaler.fit_transform(features)
    return scaled_features, true_labels

def create_plot_comparison(scaled_features, kmeans, dbscan):
    # Plot the data and cluster silhouette comparison
    fig, (ax1, ax2) = plt.subplots(
        1, 2,
        figsize=(8, 6),
        sharex=True,
        sharey=True
    )

    fig.suptitle(f"Clustering Algorithm Comparison: Crescents", fontsize=16)
    fte_colors = {
        0: "#008fd5",
        1: "#fc4f30",
    }
    # The k-means plot
    km_colors = [fte_colors[label] for label in kmeans.labels_]
    ax1.scatter(scaled_features[:, 0], scaled_features[:, 1], c=km_colors)

    kmeans_silhouette = silhouette_score(
        scaled_features,
        kmeans.labels_
    ).round(2)

    ax1.set_title(
        f"k-means\nSilhouette: {kmeans_silhouette}", fontdict={"fontsize": 12}
    )

    # The dbscan plot
    db_colors = [fte_colors[label] for label in dbscan.labels_]
    ax2.scatter(scaled_features[:, 0], scaled_features[:, 1], c=db_colors)

    dbscan_silhouette = silhouette_score(
        scaled_features,
        dbscan.labels_
    ).round(2)

    ax2.set_title(
        f"DBSCAN\nSilhouette: {dbscan_silhouette}", fontdict={"fontsize": 12}
    )
    plt.savefig("kmeans_db_comparison.png")

    print("kmeans_silhouette: ", kmeans_silhouette)
    print("dbscan_silhouette: ", dbscan_silhouette)

    # comparison shows that silhouette metric is not feasible for all cases, need someting better (especially since we know the truth label)


def kmeans_eval():
    scaled_features, true_labels = create_moon_data()


    kmeans = KMeans(n_clusters=2)
    dbscan = DBSCAN(eps=0.3)

    kmeans.fit(scaled_features)
    dbscan.fit(scaled_features)

    create_plot_comparison(scaled_features, kmeans, dbscan)

    # adjusted rand index (ARI) - compare true cluster assignments and predicted labels
    # ari close to 0 -> random, ari close to 1 -> perfect clustering
    ari_kmeans = adjusted_rand_score(true_labels, kmeans.labels_)
    ari_dbscan = adjusted_rand_score(true_labels, dbscan.labels_)

    print("kmeans_ari: ", ari_kmeans)
    print("dbscan_are: ", ari_dbscan)

if __name__ == "__main__":
    kmeans_eval()
