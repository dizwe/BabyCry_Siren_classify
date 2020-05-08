class BasePredictor(object):
    def __init__(self, model, data, config):
        self.model = model
        self.data = data
        self.config = config

    def predict(self):
        raise NotImplementedError
