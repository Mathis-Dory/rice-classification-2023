import ssl
import sys

import numpy as np
from matplotlib import pyplot as plt

# Do nut trunc print
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

from utils import load_data, show_samples

np.set_printoptions(threshold=sys.maxsize)
# Following line because import of VGG16 failed if I do not use it
# https://stackoverflow.com/questions/47231408/downloading-resnet50-in-keras-generates-ssl-certificate-verify-failed
ssl._create_default_https_context = ssl._create_unverified_context
plt.rcParams.update({"font.size": 17})


def main():
    dataset = PrepareDataset("./dataset/")
    x, y, df = load_data()
    show_samples(df)
    x_train, x_test, y_train, y_test = dataset.split(x, y)
    x_train_features, x_test_features, process = dataset.exctract_features(
        x_train, x_test, "pca"
    )
    x_test_indexes = dataset.x_test_indexes
    kmean = KmeansModel(
        x_train_features,
        x_test_features,
        y_train,
        y_test,
        df["filepath"],
        x_test_indexes,
    )
    # I chose 5 because I already Know I have 5 labels
    kmean.fit_predict(5, process)


class PrepareDataset:
    def __init__(self, path_dataset_folder):
        self.data_folder = path_dataset_folder

    def split(self, x, y):
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
        self.x_test_indexes = [i for i, x in enumerate(x_test)]
        return x_train, x_test, y_train, y_test

    def exctract_features(self, x_train, x_test, process):
        if process == "pca":

            pca = PCA(n_components=100)
            x_train = x_train.reshape(
                -1, x_train.shape[1] * x_train.shape[2] * x_train.shape[3]
            )

            x_test = x_test.reshape(
                -1, x_test.shape[1] * x_test.shape[2] * x_test.shape[3]
            )
            pca.fit(x_train)
            x_train = pca.transform(x_train)
            x_test = pca.transform(x_test)
            return x_train, x_test, pca


class KmeansModel:
    def __init__(
        self,
        x_train_features,
        x_test_features,
        y_train,
        y_test,
        image_paths,
        x_test_indexes,
    ):
        self.x_train = x_train_features
        self.x_test = x_test_features
        self.y_train = y_train
        self.y_test = y_test
        self.image_paths = image_paths
        self.x_test_indexes = x_test_indexes

    def fit_predict(self, n, process):
        kmeans = MiniBatchKMeans(n_clusters=n, batch_size=1000, random_state=42)
        kmeans.fit(self.x_train)
        pred_labels = kmeans.predict(self.x_test)

        # Create an array that contains 5 other arrays (one per cluster)
        clusters = [[] for _ in range(n)]
        # For each cluster append all index of pred_labels where the predicted label is equal to the current cluster
        for i, label in enumerate(pred_labels):
            clusters[label].append(i)

        # for each cluster
        # Plot the first 50 images of each cluster
        for cluster in clusters:
            fig, axes = plt.subplots(4, 10, figsize=(30, 30))
            # Loop 40 first images
            for j, idx in enumerate(cluster[:40]):
                # Find original image among test using the index from the predicted labels we saved
                img = plt.imread(self.image_paths[self.x_test_indexes[idx]])
                ax = axes[j // 10, j % 10]
                ax.imshow(img)
                # Get the true label of the image
                ax.set_title(self.y_test[idx])
            fig.tight_layout()
            plt.show()


if __name__ == "__main__":
    main()
