from flask import Flask
from utils.config import process_config
from utils.dirs import create_dirs
from utils.args import get_args
from utils import factory
import sys
import librosa
import numpy as np

app = Flask(__name__)

def audio_norm(data):
    max_data = np.max(data)
    min_data = np.min(data)
    data = (data-min_data)/(max_data-min_data+1e-6)
    return data-0.5

config = process_config('configs/wav_classify_config.json')


def load_model():
    global model
    model = factory.create("models."+config.model.name)(config)
    model = model.model
    model.load_weights(config.predictor.predict_model)
    
    model.compile(optimizer=config.model.optimizer,
                    loss='categorical_crossentropy',
                    metrics=['acc'])
    model._make_predict_function()
    # global graph
    # graph = tf.get_default_graph()

def prepare_data(file_path):
    amp, sr = librosa.load(file_path,sr=8000)
    amp = audio_norm(amp) # normalize
    if amp.shape[0]<config.model.duration*sr:
        # 왼쪽 오른쪽 똑같은 크기로 reflect  
        amp=np.pad(amp,int(np.ceil((10*sr-amp.shape[0])/2)),mode='reflect')
    amp=amp[:config.model.duration*sr]  
    data = np.expand_dims(amp, axis=0)
    data = np.expand_dims(data, axis=-1)
    return data

def temp_predict(data):
    with graph.as_default():
        return model.predict(data)


@app.route('/')
def hello_world():
    # capture the config path from the run arguments
    # then process the json configuration fill
    # try:
    print(111)
    # args = get_args()
    # config = process_config(args.config)
    data = prepare_data(config.predictor.predict_file_path)

    
    print('Create the model.')

    # predictor = factory.create("predictors."+config.predictor.name)(model.model, data, config)
    # predictor.predict()
    class_list = ['baby cry', 'siren', 'etc']

    print("-----------predict Result-------------------") 
    print(data.shape) 
    
    #https://github.com/keras-team/keras/issues/6462
    #https://github.com/keras-team/keras/issues/6124
    predict_result =  model.predict(data)
    print(predict_result)
    print(class_list[np.argmax(predict_result)])
    return "111"

    # except Exception as e:
    #      print(e)

# python -m flask run 하면 이게 실행이 안될수도 있다.!
# python hello.py 하고 이걸 하면 된다!
if __name__ == "__main__":
    print(("* Loading Keras model and Flask starting server..."
        "please wait until server has fully started"))
    load_model()
    app.run()