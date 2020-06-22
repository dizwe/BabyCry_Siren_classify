from comet_ml import Experiment
from utils.config import process_config
from utils.dirs import create_dirs
from utils.args import get_args
from utils import factory
import sys
from sklearn.metrics import classification_report
import numpy as np

def main():
    # capture the config path from the run arguments
    # then process the json configuration fill
    try:
        args = get_args()
        config = process_config(args.config)

        # create the experiments dirs
        create_dirs([config.callbacks.tensorboard_log_dir, config.callbacks.checkpoint_dir])
        
        print('Create the data generator.')
        data_loader = factory.create("data_loader."+config.data_loader.name)(config)

        print('Create the model.')
        model = factory.create("models."+config.model.name)(config)
        
        for i in range(config.data_loader.kfold):
            
            print('Create the trainer')
            
            trainer = factory.create("trainers."+config.trainer.name)(model.model, data_loader.get_train_data(i), config)

            print('Start training the model.')
            trainer.train()
            
            # evaluator = factory.create("evaluators."+config.evaluator.name)(model.model, data_loader.get_test_data(), config)
            # evaluator.evaluate()
            evaluator = factory.create("evaluators."+config.evaluator.name)(model.model, data_loader.get_test_data(i), config)
            
            result, y = evaluator.evaluate()
            result_idx = np.argmax(result,axis=1)
            y_idx = np.argmax(y,axis=1)
            print(classification_report(result_idx, y_idx))
    except Exception as e:
        print(e)
        sys.exit(1)

if __name__ == '__main__':
    main()
