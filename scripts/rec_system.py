class RecommendSystem(object):
    """
    Recommend system, a framework wrapping algorithm. A system may contain more than one algorithm, e.g. model ensemble.
    If there is only one algorithm in the system, the `train, evaluate, predict` method are basically the same as the algorithm.
    """
    def __init__(self):
        pass

    def train(self, dataset):
        """
        dataset: training dataset, the structure varies across algorithms.
        """
        pass

    def evaluate(self, dataset):
        """
        return evaluations based on metrics
        """
        pass

    def predict(self, dataset):
        """
        return predictions
        """
        pass