from util.ml_model import MLModel
import numpy as np


class NearestNeighborClassifier(MLModel):
    def __init__(self, **kwargs):
        self.k = kwargs.get('k', 1)
        self.p = kwargs.get('p', 2)
        self.train_data = None
        self.train_labels = None

    def fit(self, data: np.ndarray, labels: np.ndarray):
        self.train_data = data
        self.train_labels = labels

    def predict(self, data: np.ndarray):
        # Difference tensor has train points on first axis,
        # test points on second axis and point coordinates on third axis
        # Entry (i, j, k) is the kth coordinate of the difference between the ith train point and the jth test point
        # Broadcasting (m_train, 1, n) - (m_test, n) -> (m_train, m_test, n)
        diff = self.train_data[:, np.newaxis] - data
        # Apply norm over third axis
        distances = np.power(np.sum(np.power(np.abs(diff), self.p), axis=2), 1 / self.p)
        # Sort indices of every row
        sorted_ind = np.argsort(distances, axis=0)
        # Tile labels over columns to match for train data
        tiled_labels = np.tile(self.train_labels, (data.shape[0], 1)).T
        # Take top k labels, sorted by distance
        k_neighbors = np.take_along_axis(tiled_labels, sorted_ind, axis=0)[:self.k, :]
        unique = np.unique(self.train_labels)  # Find unique labels
        # Count labels for every test point
        neighbor_count = np.apply_along_axis(np.bincount, 0, k_neighbors, minlength=unique.shape[0])
        k_neighbor_count = np.take_along_axis(neighbor_count, k_neighbors, axis=0)
        # Find most frequent label(s) for every test point
        most_frequent = np.argmax(k_neighbor_count, axis=0)
        # Tie-break using closest neighbor
        predictions = k_neighbors[most_frequent, np.arange(k_neighbors.shape[1])]
        return predictions
