class AbstractRecommender:
    def __init__(self):
        """

        Initialize model configuration.

        """
        pass

    def build_graph(self):
        """

        Build tensorflow computational graph.

        """
        # Placeholder

        # Network structure

        # Inference

        # Loss & optimization

        raise NotImplementedError

    def train_model(self):
        """
        Execute train loop for a single epoch given dataset and train options.

        """
        raise NotImplementedError

    def predict(self, dataset):
        """
        Make prediction on test data which is stored in dataset.
        Depending on the task, returning dictionary should contain different values.
        - Rating prediction: Predicted ratings on test user-item.
        - Item ranking: Predicted scores on all negative items per user.
        - Sequence: ?
        """
        raise NotImplementedError