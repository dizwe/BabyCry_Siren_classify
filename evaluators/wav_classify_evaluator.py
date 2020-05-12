from base.base_evaluator import BaseEvaluator
import os

class WavClassifyEvaluator(BaseEvaluator):
    def __init__(self, model, data, config):
        super(WavClassifyEvaluator, self).__init__(model, data, config)

    def evaluate(self):
        if self.config.evaluator.evaluate_model != "":
            self.model.load_weights(self.config.evaluator.evaluate_model)
            self.model.compile(optimizer=self.config.model.optimizer,
                            loss='categorical_crossentropy',
                            metrics=['acc'])
        # print("-----------Evaluate Result-------------------")
        # print(self.model.evaluate(x= self.data[0], y=self.data[1]))
        result = self.model.predict(self.data[0])
        return result, self.data[1]
        
