import os
import ssl
import sys

import numpy as np
import pandas as pd
import seaborn as sb
from keras.applications import ResNet50
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot as plt

# Do nut trunc print
from PIL import Image
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.metrics import calinski_harabasz_score, silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle

np.set_printoptions(threshold=sys.maxsize)
# Following line because import of VGG16 failed if I do not use it
# https://stackoverflow.com/questions/47231408/downloading-resnet50-in-keras-generates-ssl-certificate-verify-failed
ssl._create_default_https_context = ssl._create_unverified_context


def main():
    data_path = PrepareDataset("./dataset/")
    data = data_path.load()
    data_path.show_samples(data)
    features = data_path.exctract_features(data)
    features = data_path.preprocess_features(features)
    x_train, x_test, y_train, y_test = data_path.split(features, data)
    kmean = KmeansModel(x_train, x_test, y_train, y_test)
    trained_kmean = kmean.train_model(5)
    kmean.predict_evaluate(trained_kmean)


class PrepareDataset:
    def __init__(self, path_dataset_folder):
        self.data_folder = path_dataset_folder
        self.model_resnet = ResNet50(
            weights="imagenet",
            include_top=False,
        )

    def load(self):
        images_list = []
        labels = []
        # Get all folders available
        for item in os.scandir(self.data_folder):
            # Take only folders (because my images are splitted into 5 different folders)
            if item.is_dir():
                # Foreach image
                for image in os.scandir(f"{self.data_folder}/{item.name}"):
                    # Add the label of the image
                    labels.append(item.name)
                    # Prepare a dataframe with the filepath of the images
                    images_list.append(image.path)
        df = pd.DataFrame()
        df["filepath"] = images_list
        df["label"] = labels
        df = shuffle(df)
        df.reset_index(drop=True, inplace=True)
        df.head(5)
        # Display proportion of labels
        sb.countplot(x=labels)
        plt.show()
        return df

    def show_samples(self, df):
        plt.figure(figsize=(10, 10))
        for i, _ in enumerate(range(1, 16), start=1):
            if i in [1, 6, 11]:
                label = "Karacadag"
            elif i in [2, 7, 12]:
                label = "Ipsala"
            elif i in [3, 8, 13]:
                label = "Basmati"
            elif i in [4, 9, 14]:
                label = "Arborio"
            else:
                label = "Jasmine"
            plt.subplot(3, 5, i)
            # Take random sample corresponding to a specific label
            random_row = df[df["label"] == label].sample()
            sample = Image.open(random_row["filepath"].values[0])
            plt.title(label)
            plt.imshow(sample)
        plt.show()

    def exctract_features(self, df):
        generator = ImageDataGenerator(
            rescale=1.0 / 255,
        )
        data = generator.flow_from_dataframe(
            dataframe=df,
            target_size=(64, 64),
            batch_size=64,
            x_col="filepath",
            y_col="label",
            class_mode=None,
            shuffle=False,
        )
        features = self.model_resnet.predict(data)
        features = features.reshape(features.shape[0], -1)
        return features

    def preprocess_features(self, features):
        scaler = MinMaxScaler()
        features = scaler.fit_transform(features)
        pca = PCA(n_components=100)
        features = pca.fit_transform(features)
        return features

    def split(self, features, df):
        x_train, x_test, y_train, y_test = train_test_split(
            features, df["label"], test_size=0.3
        )
        return x_train, x_test, y_train, y_test


class KmeansModel:
    def __init__(self, x_train, x_test, y_train, y_test):
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test

    def train_model(self, n):
        kmeans = MiniBatchKMeans(n_clusters=n)
        return kmeans.fit(self.x_train)

    def predict_evaluate(self, trained_kmeans):
        y_pred = trained_kmeans.predict(self.x_test)

        silhouette = silhouette_score(self.x_test, y_pred)
        print("Silhouette score:", silhouette)
        calinski = calinski_harabasz_score(self.x_test, y_pred)
        print("Calinski_harabasz score:", calinski)


if __name__ == "__main__":
    main()
