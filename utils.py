import os
from itertools import islice

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sb
from PIL import Image


def load_data(path="./dataset"):
    """
    Can be used to load the dataset
    """

    X = []
    y = []
    images_path = []
    # Get all folders available
    for item in os.scandir(path):
        # Take only folders (because my images are splitted into 5 different folders) and ignore files
        if item.is_dir():
            # Foreach image but take only the first 1500 because lack of memory
            for image in islice(os.scandir(f"{path}/{item.name}"), 1000):
                # Add the label of the image
                if item.name == "Arborio":
                    y.append(0)
                elif item.name == "Basmati":
                    y.append(1)
                elif item.name == "Ipsala":
                    y.append(2)
                elif item.name == "Jasmine":
                    y.append(3)
                else:
                    y.append(4)
                images_path.append(image.path)
                temp = plt.imread(image.path)
                X.append(temp)

    df = pd.DataFrame()
    df["filepath"] = images_path
    df["label"] = y
    # normalize it
    X = np.asarray(X).reshape(-1, 250, 250, 3) / 255.0
    # Display proportion of labels
    sb.countplot(x=y)
    plt.savefig("./images/dataset/dataset.png", dpi=75, format="png")
    plt.show()

    return X, np.asarray(y), df


def show_samples(df):
    plt.figure(figsize=(10, 10))
    for i, _ in enumerate(range(1, 16), start=1):
        if i in [1, 6, 11]:
            title = "Arborio"
            label = 0
        elif i in [2, 7, 12]:
            title = "Basmati"
            label = 1
        elif i in [3, 8, 13]:
            title = "Ipsala"
            label = 2
        elif i in [4, 9, 14]:
            label = 3
            title = "Jasmine"
        else:
            title = "Karacadag"
            label = 4
        plt.subplot(3, 5, i)
        # Take random sample corresponding to a specific label
        random_row = df[df["label"] == label].sample()
        sample = Image.open(random_row["filepath"].values[0])
        plt.title(title)
        plt.savefig("./images/dataset/samples.png", dpi=75, format="png")
        plt.imshow(sample)
    plt.show()
