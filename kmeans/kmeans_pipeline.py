import tarfile
import urllib

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

def download_and_extract():
    """
    download data to be used
    """
    uci_tcga_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00401/"
    archive_name = "TCGA-PANCAN-HiSeq-801x20531.tar.gz"

    # Build the url
    full_download_url = urllib.parse.urljoin(uci_tcga_url, archive_name)

    # Download the file
    r = urllib.request.urlretrieve (full_download_url, archive_name)

    # Extract the data from the archive
    tar = tarfile.open(archive_name, "r:gz")
    tar.extractall()
    tar.close()

def load_data():
    """
    load data as numpy arrays from downloaded file
    """
    datafile = "TCGA-PANCAN-HiSeq-801x20531/data.csv"
    labels_file = "TCGA-PANCAN-HiSeq-801x20531/labels.csv"

    data = np.genfromtxt(
        datafile,
        delimiter=",",
        usecols=range(1, 20532),
        skip_header=1
    )

    true_label_names = np.genfromtxt(
        labels_file,
        delimiter=",",
        usecols=(1,),
        skip_header=1,
        dtype="str"
    )

    return data, true_label_names




def main():
    #download_and_extract()
    data, true_labels = load_data()

    # encode alphanumeric labels
    label_encoder = LabelEncoder()
    true_labels = label_encoder.fit_transform(true_labels)
    n_clusters = len(label_encoder.classes_)

    # dimensional reduction using principal component analysis -> mapping on components
    n_components = 2


    # preprocessing pipeline
    preprocessor = Pipeline(
        [
            ("scaler", MinMaxScaler()),
            ("pca", PCA(n_components=n_components, random_state=42)),
        ]
    )

    # parameter tuning -> maximize performance

    # clustering pipeline
    clusterer = Pipeline(
       [
           (
               "kmeans",
               KMeans(
                   n_clusters=n_clusters,
                   init="k-means++",
                   n_init=50,
                   max_iter=500,
                   random_state=42,
               ),
           ),
       ]
    )

    # full pipeline
    pipe = Pipeline(
        [
            ("preprocessor", preprocessor),
            ("clusterer", clusterer)
        ]
    )

    # perform fit on data
    pipe.fit(data)


    # get results
    preprocessed_data = pipe["preprocessor"].transform(data)
    predicted_labels = pipe["clusterer"]["kmeans"].labels_
    silhouetteScore = silhouette_score(preprocessed_data, predicted_labels)
    print("silhouette_score: ", silhouetteScore)
    ari_score = adjusted_rand_score(true_labels, predicted_labels)
    print("ari_score: ", ari_score)

    # plotting
    if n_components == 2:

        pcadf = pd.DataFrame(
            pipe["preprocessor"].transform(data),
            columns=["component_1", "component_2"],
        )

        pcadf["predicted_cluster"] = pipe["clusterer"]["kmeans"].labels_
        pcadf["true_label"] = label_encoder.inverse_transform(true_labels)

        #plt.style.use("fivethirtyeight")
        plt.figure(figsize=(8, 8))

        scat = sns.scatterplot(
            "component_1",
            "component_2",
            s=50,
            data=pcadf,
            hue="predicted_cluster",
            style="true_label",
            palette="Set2",
        )

        scat.set_title(
            "Clustering results from TCGA Pan-Cancer\nGene Expression Data"
        )
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
        plt.tight_layout()
        plt.savefig("PlotPCAComponents.png")

    # using only two components in PCA step -> won’t capture all of the explained variance of the input data

    # Explained variance measures the discrepancy between the PCA-transformed data and the actual input data

    #  relationship between n_components and explained variance can be visualized in a plot
    # to show you how many components you need in your PCA to capture a certain percentage of the variance in the input data

    # Empty lists to hold evaluation metrics
    silhouette_scores = []
    ari_scores = []
    for n in range(2, 11):
        # This set the number of components for pca,
        # but leaves other steps unchanged
        pipe["preprocessor"]["pca"].n_components = n
        pipe.fit(data)

        silhouette_coef = silhouette_score(
            pipe["preprocessor"].transform(data),
            pipe["clusterer"]["kmeans"].labels_,
        )
        ari = adjusted_rand_score(
            true_labels,
            pipe["clusterer"]["kmeans"].labels_,
        )

        # Add metrics to their lists
        silhouette_scores.append(silhouette_coef)
        ari_scores.append(ari)

    # plotting
    #plt.style.use("fivethirtyeight")
    plt.figure(figsize=(6, 6))
    plt.plot(
        range(2, 11),
        silhouette_scores,
        c="#008fd5",
        label="Silhouette Coefficient",
    )
    plt.plot(range(2, 11), ari_scores, c="#fc4f30", label="ARI")

    plt.xlabel("n_components")
    plt.legend()
    plt.title("Clustering Performance as a Function of n_components")
    plt.tight_layout()
    plt.savefig("nComponents_vs_explainedVariance.png")

    # interpretation
    # silhoeutte coefficent decreases linearly because depends on distance of points, which increases with the dimensions
    # ari improves with copmonents and decreases for too high values



if __name__ == "__main__":
    main()
