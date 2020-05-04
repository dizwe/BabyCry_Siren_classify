from base.base_data_loader import BaseDataLoader
from sklearn.model_selection import train_test_split
import os
import librosa
import numpy as np

def audio_norm(data):
    max_data = np.max(data)
    min_data = np.min(data)
    data = (data-min_data)/(max_data-min_data+1e-6)
    return data-0.5

class WavClassifyDataLoader(BaseDataLoader):
    def __init__(self, config):
        super(WavClassifyDataLoader, self).__init__(config)
        # 원래는numpy array로 return
        a_class = 'Baby cry, infant cry'
        b_class = 'Siren'
        
        data_sec_size = 10 # 10초
        # self.X = np.empty((data_sec_size*8000,), int)
        X = []
        y = [] #np.empty((1,), int)
        errors = []
        cur_path = os.getcwd() 
        # parent_dir = os.path.abspath(os.path.join(cur_path, os.pardir))
        dataset_dir = os.path.join(cur_path, 'datasets')
        for cl_idx,cl in enumerate([a_class, b_class]):
            class_dir = os.path.join(dataset_dir, cl)
            for file_idx, a_file in enumerate(os.listdir(class_dir)):
                if a_file.endswith('.mp3'):
                    try: 
                        print(a_file)
                        amp, sr = librosa.load(os.path.join(class_dir, a_file),sr=8000)
                        amp = audio_norm(amp) # normalize
                        if amp.shape[0]<data_sec_size*sr:
                            # 왼쪽 오른쪽 똑같은 크기로 reflect
                            amp=np.pad(amp,int(np.ceil((10*sr-amp.shape[0])/2)),mode='reflect')
                        amp=amp[:data_sec_size*sr]  
                        # amp = amp.reshape((data_sec_size, sr))
                        
                        X.append(amp)
                        cl = [0,0,0]
                        cl[cl_idx] = 1  # onehot encoding
                        y.append(cl)
                    except Exception as e:
                        print(e)
                        print(a_file)
                        errors.append(a_file)

        # https://stackoverflow.com/questions/22392497/how-to-add-a-new-row-to-an-empty-numpy-array        
        # 쌓기  
        X = np.stack(X, axis=0)
        X = np.expand_dims(X, axis=-1)
        y = np.stack(y, axis=0)
        print(errors)
        # import json
        # with open('errors.json', 'w', encoding='utf-8') as make_file:
        #     json.dump(errors, make_file, indent="\t")
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.33, random_state=42)
        
    def get_train_data(self):
        return self.X_train, self.y_train

    def get_test_data(self):
        return self.X_test, self.y_test
