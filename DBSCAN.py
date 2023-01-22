import ssl

import numpy as np
import seaborn as sb
from keras.applications import InceptionResNetV2
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split

from utils import load_data, show_samples

plt.rcParams.update({"font.size": 17})

# Following line because import of pre trained model failed if I do not use it
# https://stackoverflow.com/questions/47231408/downloading-resnet50-in-keras-generates-ssl-certificate-verify-failed
ssl._create_default_https_context = ssl._create_unverified_context


PROCESS = "pca"


def main():
    dataset = PrepareDataset("./dataset/")
    x, y, df = load_data()
    show_samples(df)
    x_train, x_test, y_train, y_test, idx_test = dataset.split(x, y)
    # Choose between only pca or resnet + pca
    x_train_features, x_test_features = dataset.exctract_features(x_train, x_test)
    kmean = DBSCANModel(
        x_train_features,
        x_test_features,
        y_train,
        y_test,
        df["filepath"],
        idx_test,
    )
    predicted_labels, true_labels, model = kmean.fit_predict()
    kmean.evaluate_model(predicted_labels, true_labels, model)


class PrepareDataset:
    def __init__(self, path_dataset_folder):
        self.data_folder = path_dataset_folder

    def split(self, x, y):
        # Regular split but I also save the original indexes from the dataframe in order to be able to plot results without the features extraction
        x_train, x_test, y_train, y_test, idx_train, idx_test = train_test_split(
            x, y, range(len(x)), test_size=0.3
        )
        sb.countplot(x=y_test)
        plt.title("Split in test data")
        plt.savefig(f"./images/DBSCAN/test_split_{PROCESS}.png", dpi=75, format="png")
        plt.show()

        sb.countplot(x=y_train)
        plt.title("Split in train data")
        plt.savefig(f"./images/DBSCAN/train_split_{PROCESS}.png", dpi=75, format="png")
        plt.show()

        return x_train, x_test, y_train, y_test, idx_test

    def exctract_features(self, x_train, x_test):
        # Features extraction using pca
        if PROCESS == "pca":

            pca = PCA(n_components=50)
            # Flatten (number_samples, number_features) to bot able to use pca
            x_train = x_train.reshape(
                -1, x_train.shape[1] * x_train.shape[2] * x_train.shape[3]
            )

            x_test = x_test.reshape(
                -1, x_test.shape[1] * x_test.shape[2] * x_test.shape[3]
            )
            pca.fit(x_train)
            x_train = pca.transform(x_train)
            x_test = pca.transform(x_test)

        elif PROCESS == "resnet":
            # Load the RInceptionResNet50 model
            model = InceptionResNetV2(
                weights="imagenet", include_top=False, input_shape=(250, 250, 3)
            )
            # Extract features from the images using the ResNet50 model
            x_train = model.predict(x_train)
            x_test = model.predict(x_test)

            # Flatten the features
            x_train = x_train.reshape(x_train.shape[0], -1)
            x_test = x_test.reshape(x_test.shape[0], -1)

            # Apply t-SNE to reduce the dimensionality of the features
            tsne = TSNE(n_components=2)
            x_train = tsne.fit_transform(x_train)
            x_test = tsne.fit_transform(x_test)
        else:
            print("Please choose between pca or resnet extraction")
            return 0

        return x_train, x_test


class DBSCANModel:
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

    def fit_predict(self, eps, min_samples):
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        dbscan.fit(self.x_train)
        prediction = dbscan.fit_predict(self.x_test)
        pred_labels = prediction.labels_
        number_of_clusters = len(np.unique(pred_labels)) - (
            1 if -1 in np.unique(pred_labels) else 0
        )

        true_labels = []
        # Create an array that contains 5 other arrays (one per cluster)
        clusters = [[] for _ in range(number_of_clusters)]
        # For each cluster append all index of pred_labels where the predicted label is equal to the current cluster
        for i, label in enumerate(pred_labels):
            clusters[label].append(i)

        for i, cluster in enumerate(clusters, start=1):
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
            plt.savefig(
                f"./images/DBSCAN/prediction_sample_cluster{i}_{PROCESS}.png",
                dpi=75,
                format="png",
            )
            plt.show()
            return pred_labels, true_labels, dbscan

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
        plt.savefig(
            f"./images/DBSCAN/clusters_centroids_{PROCESS}.png", dpi=75, format="png"
        )
        plt.show()

        f, ax = plt.subplots(1, 1, figsize=(30, 30))
        metrics.ConfusionMatrixDisplay.from_predictions(
            true_labels,
            predicted_labels,
            ax=ax,
            normalize=None,
        )
        plt.title("Confusion matrix", fontsize=28)
        plt.ylabel("True label", fontsize=25)
        plt.xlabel("Predicted label", fontsize=25)
        tick_marks = np.arange(5)
        plt.xticks(tick_marks, range(1, 6))
        plt.yticks(tick_marks, range(1, 6))
        plt.savefig(
            f"./images/DBSCAN/confusion_matrix_{PROCESS}.png", dpi=75, format="png"
        )
        plt.show()


if __name__ == "__main__":
    main()
