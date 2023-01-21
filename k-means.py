import ssl
import sys

import numpy as np
from keras.applications import ResNet50
from keras.applications.resnet import preprocess_input
from matplotlib import pyplot as plt

# Do nut trunc print
from sklearn import metrics
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
    x_train, x_test, y_train, y_test, idx_test = dataset.split(x, y)
    # Choose between only pca or resnet + pca
    x_train_features, x_test_features = dataset.exctract_features(
        x_train, x_test, "resnet"
    )
    kmean = KmeansModel(
        x_train_features,
        x_test_features,
        y_train,
        y_test,
        df["filepath"],
        idx_test,
    )
    # I chose 5 because I already Know I have 5 labels
    predicted_labels, true_labels, model = kmean.fit_predict(5)
    kmean.evaluate_model(predicted_labels, true_labels, model)


class PrepareDataset:
    def __init__(self, path_dataset_folder):
        self.data_folder = path_dataset_folder

    def split(self, x, y):
        # Regular split but I also save the original indexes from the dataframe in order to be able to plot results without the features extraction
        x_train, x_test, y_train, y_test, idx_train, idx_test = train_test_split(
            x, y, range(len(x)), test_size=0.3
        )
        return x_train, x_test, y_train, y_test, idx_test

    def exctract_features(self, x_train, x_test, process):
        # Features extraction using pca
        if process == "pca":

            pca = PCA(n_components=50)
            x_train = x_train.reshape(
                -1, x_train.shape[1] * x_train.shape[2] * x_train.shape[3]
            )

            x_test = x_test.reshape(
                -1, x_test.shape[1] * x_test.shape[2] * x_test.shape[3]
            )
            pca.fit(x_train)
            x_train = pca.transform(x_train)
            x_test = pca.transform(x_test)

        elif process == "resnet":
            # Load the ResNet50 model
            model = ResNet50(
                weights="imagenet", include_top=False, input_shape=(224, 224, 3)
            )
            # Preprocess the images
            x_train = preprocess_input(x_train)
            x_test = preprocess_input(x_test)
            # Extract features from the images using the ResNet50 model
            x_train = model.predict(x_train)
            x_test = model.predict(x_test)
            # Flatten the features
            x_train = x_train.reshape(x_train.shape[0], -1)
            x_test = x_test.reshape(x_test.shape[0], -1)

            # Also apply pca here in order to be able to plot clusers and centroids after prediction because reesnet give high dimensional features
            pca = PCA(n_components=50)
            x_train = pca.fit_transform(x_train)
            x_test = pca.fit_transform(x_test)

        else:
            print("Please choose between pca or resnet extraction  ")
            return 0

        return x_train, x_test


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

    def fit_predict(self, n):
        kmeans = MiniBatchKMeans(n_clusters=n, batch_size=500, random_state=42)
        kmeans.fit(self.x_train)
        pred_labels = kmeans.predict(self.x_test)
        true_labels = []
        # Create an array that contains 5 other arrays (one per cluster)
        clusters = [[] for _ in range(n)]
        # For each cluster append all index of pred_labels where the predicted label is equal to the current cluster
        for i, label in enumerate(pred_labels):
            clusters[label].append(i)

        # for each cluster
        # Plot the first 40 images of each cluster
        i = 0
        for cluster in clusters:
            i += 1
            fig, axes = plt.subplots(4, 10, figsize=(30, 30))

            for j, idx in enumerate(cluster):
                # Find original image among test using the index from the predicted labels we saved
                # plot only 40 first images
                if j < 40:
                    img = plt.imread(self.image_paths[self.x_test_indexes[idx]])
                    ax = axes[j // 10, j % 10]
                    ax.imshow(img)
                    # Get the true label of the image
                    ax.set_title(self.y_test[idx])
                true_labels.append(self.y_test[idx])
            fig.suptitle(f"Cluster number: {i}")
            fig.tight_layout()
            plt.show()

        return pred_labels, true_labels, kmeans

    def evaluate_model(self, predicted_labels, true_labels, model):
        acc = metrics.accuracy_score(true_labels, predicted_labels)
        print("Accuracy:", acc)
        sil = metrics.silhouette_score(self.x_test, predicted_labels)
        print("Silhouette score:", sil)
        centroids = model.cluster_centers_
        u_labels = np.unique(predicted_labels)
        for i in u_labels:
            plt.scatter(
                self.x_test[predicted_labels == i, 0],
                self.x_test[predicted_labels == i, 1],
                label=i,
            )
        plt.scatter(centroids[:, 0], centroids[:, 1], s=80, color="black", marker="x")
        plt.legend()
        plt.show()


if __name__ == "__main__":
    main()
