
from base.base_predictor import BasePredictor
import os
import numpy as np

class WavClassifyPredictor(BasePredictor):
    def __init__(self, model, data, config):
        super(WavClassifyPredictor, self).__init__(model, data, config)

    def predict(self):
        class_list = ['baby cry', 'siren', 'etc']
        self.model.load_weights(self.config.predictor.predict_model)
        self.model.compile(optimizer=self.config.model.optimizer,
                        loss='categorical_crossentropy',
                        metrics=['acc'])
        print("-----------predict Result-------------------")
        predict_result = self.model.predict(x = self.data)
        print(predict_result)
        print(class_list[np.argmax(predict_result)])
        
