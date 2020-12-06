import abc


class MLModel(abc.ABC):
    @abc.abstractmethod
    def fit(self, data, labels):
        raise NotImplementedError()

    @abc.abstractmethod
    def predict(self, data):
        raise NotImplementedError()
