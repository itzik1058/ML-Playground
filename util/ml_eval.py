from util.ml_model import MLModel
import numpy as np


def accuracy(model: MLModel, data, labels):
    return np.mean(model.predict(data) == labels)
