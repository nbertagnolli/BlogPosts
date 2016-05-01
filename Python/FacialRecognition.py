__author__ = 'tetracycline'
__author__ = 'tetracycline'
from scipy import misc
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.decomposition import PCA
from sklearn.learning_curve import learning_curve
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
import os
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression


def find_label(name):
    """
    This method extracts the label of the individual in the yale face data set.

    Args:
        :param name: (string) The file name for which we want to extract the face label

    Returns:
        :return: (Int) the integer label of the subject in the yale face data set
    """
    return int(name.split(".")[0].split("t")[1])


def learning_curve_mod(data, labels, clf, percents, d=100, avg=3, test_size=.2):
    """
    This method calculates the performance of the training and cross validation test set as the training
    set size increases and returns the performance at each percent

    Args:
        :param data: (md.array) The raw data to use for training and cross validation testing
        :param labels: (nd.array) the labels associated with the data
        :param clf: (sklearn classifier) the classifier to be used for training
        :param percents: (nd.array) a list of percent of training data to use
        :param d:  (int) The number of principle components to calculate
        :param avg: (int) The number of iterations to average when calculating performance
        :param test_size: (double [0,1]) The size of the testing set

    Return:
        :return: train_accuracies (list) performance on the training set
        :return: test_accuracies (list) performance on the testing set
    """
    # split into train and testing dataset
    x_train, x_test, y_train, y_test = train_test_split(data.T, labels, test_size=test_size, random_state=0)
    x_test = x_test.T
    train_accuracies = []
    test_accuracies = []
    for percent in percents:
        temp_train_accuracies = []
        temp_test_accuracies = []
        print percent
        for i in range(0, avg):
            x_train_2, x_test_2, y_train_2, y_test_2 = train_test_split(x_train, y_train, test_size=percent)
            x_train_2 = x_train_2.T

            # Subtract off the mean
            mean_face = np.mean(x_train_2, axis=1)
            x_train_2 = x_train_2 - mean_face

            # Find low dimensional subspace using PCA
            pca = PCA(n_components=d)
            pca.fit(x_train_2)
            model = pca.transform(x_train_2)

            # Project the known faces onto the face space
            label_map = np.dot(x_train_2.T, model)

            # Train a KNN classifier
            clf.fit(label_map, y_train_2)

            # project the unknown faces onto face space
            W_train = np.dot(x_train_2.T - mean_face.T, model)
            W_test = np.dot(x_test.T - mean_face.T, model)


            test_prediction = clf.predict(W_test)
            temp_test_accuracies.append(metrics.precision_score(y_test, test_prediction))
            train_prediction = clf.predict(W_train)
            temp_train_accuracies.append(metrics.precision_score(y_train_2, train_prediction))

        train_accuracies.append(np.mean(temp_train_accuracies))
        test_accuracies.append(np.mean(temp_test_accuracies))

    return train_accuracies, test_accuracies



if __name__ == "__main__":

    # Import data
    directory = "/Users/tetracycline/Data/yalefaces"
    files = os.listdir(directory)

    # Create Label vector
    labels = []
    for file in files:
        if not file == ".DS_Store":
            labels.append(find_label(file))

    labels = np.array(labels)

    # pull the first image to initialize the matrix
    img = misc.imread(directory + "/" + files[1])
    del files[1]
    faces = np.matrix(img.flatten("F")).T

    # load in all images, vectorize them, and compile into a matrix
    for file in files:
        if not file == ".DS_Store":
            img = misc.imread(directory + "/" + file)
            faces = np.concatenate([faces, np.matrix(img.flatten("F")).T], axis=1)

    # Split into training and test set
    x_train, x_test, y_train, y_test = train_test_split(faces.T, labels, test_size=0.2)
    x_train = x_train.T
    x_test = x_test.T

    # Train with knn=1 and dimensionality of 34
    d = 34
    k = 1

    # Subtract off the mean
    mean_face = np.mean(x_train, axis=1)
    x_train = x_train - mean_face

    # Find low dimensional subspace using PCA
    pca = PCA(n_components=d)
    pca.fit(x_train)
    model = pca.transform(x_train)

    # Project the known faces onto the face space
    label_map = np.dot(x_train.T, model)

    # Train a KNN classifier
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(label_map, y_train)

    # project the unknown faces onto face space
    W = np.dot(x_test.T - mean_face.T, model)

    # Predict the unknown faces
    print "precision test: ", metrics.precision_score(y_test, knn.predict(W))

    # ======================================================================================================
    # Learning curve on initial machine learning k=1, d=32
    # ======================================================================================================
    percents = np.linspace(0, .6, 15)[::-1]  # backwards
    d = 32
    k = 1
    clf = KNeighborsClassifier(n_neighbors=k)
    train_accuracies, test_accuracies = learning_curve_mod(x_train, y_train, clf, percents, d=d)
    test_plot = plt.plot(1 - percents, test_accuracies, label="Test")
    train_plot = plt.plot(1 - percents, train_accuracies, 'g-', label="Train")
    plt.xlabel("Percent of training data used")
    plt.ylabel("Model Precision")
    plt.title("Learning Curve (KNN) d=" + str(d) + ", k=" + str(k))
    plt.ylim([0, 1.1])
    plt.legend(loc=4, markerscale=2, fontsize=20)
    plt.show()


    # ======================================================================================================
    # Learning curve k=2, d=32
    # ======================================================================================================
    d = 32
    k = 2
    clf = KNeighborsClassifier(n_neighbors=k)
    train_accuracies, test_accuracies = learning_curve_mod(x_train, y_train, clf, percents, d=d)
    test_plot = plt.plot(1 - percents, test_accuracies, label="Test")
    train_plot = plt.plot(1 - percents, train_accuracies, 'g-', label="Train")
    plt.xlabel("Percent of training data used")
    plt.ylabel("Model Precision")
    plt.title("Learning Curve (KNN) d=" + str(d) + ", k=" + str(k))
    plt.ylim([0, 1.1])
    plt.legend(loc=4, markerscale=2, fontsize=20)
    plt.show()


    # ======================================================================================================
    # Learning curve k=2, d=42
    # ======================================================================================================
    d = 42
    k = 2
    clf = KNeighborsClassifier(n_neighbors=k)
    train_accuracies, test_accuracies = learning_curve_mod(x_train, y_train, clf, percents, d=d)
    test_plot = plt.plot(1 - percents, test_accuracies, label="Test")
    train_plot = plt.plot(1 - percents, train_accuracies, 'g-', label="Train")
    plt.xlabel("Percent of training data used")
    plt.ylabel("Model Precision")
    plt.title("Learning Curve (KNN) d=" + str(d) + ", k=" + str(k))
    plt.ylim([0, 1.1])
    plt.legend(loc=4, markerscale=2, fontsize=20)
    plt.show()

    # ======================================================================================================
    # Learning curve logistic regression C=10, d=32
    # ======================================================================================================
    d = 32
    C = 10
    clf = LogisticRegression(C=C)
    train_accuracies, test_accuracies = learning_curve_mod(x_train, y_train, clf, percents, d=d)
    test_plot = plt.plot(1 - percents, test_accuracies, label="Test")
    train_plot = plt.plot(1 - percents, train_accuracies, 'g-', label="Train")
    plt.xlabel("Percent of training data used")
    plt.ylabel("Model Precision")
    plt.title("Learning Curve (Logistic Regression) d=" + str(d) + ", C=" + str(C))
    plt.ylim([0, 1.1])
    plt.legend(loc=4, markerscale=2, fontsize=20)
    plt.show()

        # ======================================================================================================
    # Learning curve logistic regression C=10, d=32
    # ======================================================================================================
    d = 42
    C = 10
    clf = LogisticRegression(C=C)
    train_accuracies, test_accuracies = learning_curve_mod(x_train, y_train, clf, percents, d=d)
    test_plot = plt.plot(1 - percents, test_accuracies, label="Test")
    train_plot = plt.plot(1 - percents, train_accuracies, 'g-', label="Train")
    plt.xlabel("Percent of training data used")
    plt.ylabel("Model Precision")
    plt.title("Learning Curve (Logistic Regression) d=" + str(d) + ", C=" + str(C))
    plt.ylim([0, 1.1])
    plt.legend(loc=4, markerscale=2, fontsize=20)
    plt.show()


