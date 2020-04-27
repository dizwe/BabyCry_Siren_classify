from base.base_data_loader import BaseDataLoader
from keras.datasets import mnist
import os
import librosa
import numpy as np
class WavClassifyDataLoader(BaseDataLoader):
    def __init__(self, config):
        super(WavClassifyDataLoader, self).__init__(config)
        # 원래는numpy array로 return
        a_class = 'Baby cry, infant cry'
        b_class = 'Siren'
        
        data_sec_size = 10 # 10초
        # self.X = np.empty((data_sec_size*8000,), int)
        self.X = []
        self.y = [] #np.empty((1,), int)
        cur_path = os.getcwd() 
        # parent_dir = os.path.abspath(os.path.join(cur_path, os.pardir))
        dataset_dir = os.path.join(cur_path, 'datasets')
        for cl_idx,cl in enumerate([a_class, b_class]):
            class_dir = os.path.join(dataset_dir, cl)
            for file_idx, a_file in enumerate(os.listdir(class_dir)):
                # if file_idx ==2:
                #     break
                if a_file.endswith('.mp3'):
                    print(os.path.join(class_dir, a_file))
                    amp, sr = librosa.load(os.path.join(class_dir, a_file),sr=8000)
                    if amp.shape[0]<data_sec_size*sr:
                        # 왼쪽 오른쪽 똑같은 크기로 reflect
                        amp=np.pad(amp,int(np.ceil((10*sr-amp.shape[0])/2)),mode='reflect')
                    amp=amp[:data_sec_size*sr]  
                    print(amp.shape)
                    
                    self.X.append(amp)
                    self.y.append(cl_idx)

                    print(self.X)
                    print(self.y)
                    
        # https://stackoverflow.com/questions/22392497/how-to-add-a-new-row-to-an-empty-numpy-array        
        self.X = np.stack(self.X, axis=0)
        self.y = np.stack(self.y, axis=0)
        print(self.X.shape)
        print(self.y.shape)

    def get_train_data(self):
        return self.X_train, self.y_train

    def get_test_data(self):
        return self.X_test, self.y_test
