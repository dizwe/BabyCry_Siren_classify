from base.base_model import BaseModel
from keras.models import Sequential
from keras.layers import Input, Dense, Dropout
from keras.layers import Convolution1D, GlobalMaxPool1D, MaxPool1D


class WavClassifyModel(BaseModel):
    def __init__(self, config):
        super(WavClassifyModel, self).__init__(config)
        self.build_model()

    def build_model(self):    
        self.model = Sequential()
        self.model.add(Convolution1D(16, 9, activation='relu', padding="valid",input_shape=(self.config.model.duration* self.config.model.sampling_rate,1)))
        self.model.add(Convolution1D(16, 9, activation='relu', padding="valid"))
        self.model.add(MaxPool1D(16))
        self.model.add(Dropout(rate=0.1))

        self.model.add(Convolution1D(32, 3, activation='relu', padding="valid"))
        self.model.add(Convolution1D(32, 3, activation='relu', padding="valid"))
        self.model.add(MaxPool1D(4))
        self.model.add(Dropout(rate=0.1))

        self.model.add(Convolution1D(32, 3, activation='relu', padding="valid"))
        self.model.add(Convolution1D(32, 3, activation='relu', padding="valid"))
        self.model.add(MaxPool1D(4))
        self.model.add(Dropout(rate=0.1))
    
        self.model.add(Convolution1D(256, 3, activation='relu', padding="valid"))
        self.model.add(Convolution1D(256, 3, activation='relu', padding="valid"))
        self.model.add(GlobalMaxPool1D())
        self.model.add(Dropout(rate=0.2))

        self.model.add(Dense(64, activation='relu'))
        self.model.add(Dense(1028, activation='relu'))
        self.model.add(Dense(self.config.model.num_class, activation='softmax'))


        self.model.compile(optimizer=self.config.model.optimizer,
                             loss='categorical_crossentropy',
                             metrics=['acc'])
        


