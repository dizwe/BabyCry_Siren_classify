class BaseEvaluator(object):
    def __init__(self, model, data, config):
        self.model = model
        self.data = data
        self.config = config

    def evaluate(self):
        raise NotImplementedError
