class Algorithm(object):
    """
    Can be a neural network, statistical model, or mathematical strategy. More than just a model
    """
    def __init__(self):
        pass

    def build_dataset(self):
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


class AlgoLR(Algorithm):
    def __init__(self):
        super().__init__()

    def build_dataset(self):
        """
        data format: json
        structure:[{
            user_id: XXX,
            user_meta: {},
            item_list: [
                {
                    item_id: XXX,
                    item_meta:{}
                }
            ]
        }]
        """
        # build torch dataset
        pass