from utils.config import process_config
from utils.dirs import create_dirs
from utils.args import get_args
from utils import factory
import sys
import librosa
import numpy as np

def audio_norm(data):
    max_data = np.max(data)
    min_data = np.min(data)
    data = (data-min_data)/(max_data-min_data+1e-6)
    return data-0.5

def main():
    # capture the config path from the run arguments
    # then process the json configuration fill
    try:
        args = get_args()
        config = process_config(args.config)

        amp, sr = librosa.load(config.predictor.predict_file_path,sr=8000)
        amp = audio_norm(amp) # normalize
        if amp.shape[0]<config.model.duration*sr:
            # 왼쪽 오른쪽 똑같은 크기로 reflect  
            amp=np.pad(amp,int(np.ceil((10*sr-amp.shape[0])/2)),mode='reflect')
        amp=amp[:config.model.duration*sr]  
        data = np.expand_dims(amp, axis=0)
        data = np.expand_dims(data, axis=-1)

        print('Create the model.')
        model = factory.create("models."+config.model.name)(config)

        predictor = factory.create("predictors."+config.predictor.name)(model.model, data, config)
        predictor.predict()
        sys.stdout.flush()
    except Exception as e:
         print(e)
         sys.stdout.flush()
    #     sys.exit(1)

if __name__ == '__main__':
    main()
