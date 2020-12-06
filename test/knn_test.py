from KNN.knn import NearestNeighborClassifier
from util.ml_eval import accuracy
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


def test():
    knn = NearestNeighborClassifier(p=3, k=3)
    iris = load_iris(return_X_y=True)
    train_data, test_data, train_labels, test_labels = train_test_split(*iris, test_size=0.2, random_state=0)
    knn.fit(train_data, train_labels)
    print('Train accuracy', accuracy(knn, train_data, train_labels))
    print('Test accuracy', accuracy(knn, test_data, test_labels))


if __name__ == '__main__':
    test()
