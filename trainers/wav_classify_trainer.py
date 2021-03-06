from base.base_trainer import BaseTrain
import os
from keras.callbacks import ModelCheckpoint, TensorBoard
import matplotlib.pyplot as plt

class WavClassifyTrainer(BaseTrain):
    def __init__(self, model, data, config):
        super(WavClassifyTrainer, self).__init__(model, data, config)
        self.callbacks = []
        self.loss = []
        self.acc = []
        self.val_loss = []
        self.val_acc = []
        self.init_callbacks()

    def init_callbacks(self):
        self.callbacks.append(
            ModelCheckpoint(
                filepath=os.path.join(self.config.callbacks.checkpoint_dir, '%s-{epoch:02d}-{val_loss:.2f}.hdf5' % self.config.exp.name),
                monitor=self.config.callbacks.checkpoint_monitor,
                mode=self.config.callbacks.checkpoint_mode,
                save_best_only=self.config.callbacks.checkpoint_save_best_only,
                save_weights_only=self.config.callbacks.checkpoint_save_weights_only,
                verbose=self.config.callbacks.checkpoint_verbose,
            )
        )

        self.callbacks.append(
            TensorBoard(
                log_dir=self.config.callbacks.tensorboard_log_dir,
                write_graph=self.config.callbacks.tensorboard_write_graph,
            )
        )

        if hasattr(self.config,"comet_api_key"):
            from comet_ml import Experiment
            experiment = Experiment(api_key=self.config.comet_api_key, project_name=self.config.exp_name)
            experiment.disable_mp()
            experiment.log_parameters(self.config)
            self.callbacks.append(experiment.get_keras_callback())

    def train(self):
        hist = self.model.fit(
            self.data[0], self.data[1],
            epochs=self.config.trainer.num_epochs,
            verbose=self.config.trainer.verbose_training,
            batch_size=self.config.trainer.batch_size,
            validation_split=self.config.trainer.validation_split,
            callbacks=self.callbacks,
        )
        
        fig, acc_ax = plt.subplots() # fig, loss_ax = plt.subplots()
        # acc_ax = loss_ax.twinx() # X 
        # axis는 공유하지만, y axis는 공유하지 않는 metric

        # loss_ax.plot(hist.history['loss'], 'y', label='train loss')
        # loss_ax.plot(hist.history['val_loss'], 'r', label='val loss')

        acc_ax.plot(hist.history['acc'], 'b', label='train acc')
        acc_ax.plot(hist.history['val_acc'], 'g', label='val acc')

        acc_ax.set_xlabel('epoch')# loss_ax.set_xlabel('epoch')
        # loss_ax.set_ylabel('loss')
        acc_ax.set_ylabel('accuray')

        # loss_ax.legend(loc='upper left')
        acc_ax.legend(loc='upper left')# acc_ax.legend(loc='lower left')

        plt.savefig('fig1.png', dpi=300)

        ########### draw loss_ax
        fig, acc_ax = plt.subplots() # fig, loss_ax = plt.subplots()
        # acc_ax = loss_ax.twinx() # X 
        # axis는 공유하지만, y axis는 공유하지 않는 metric

        # loss_ax.plot(hist.history['loss'], 'y', label='train loss')
        # loss_ax.plot(hist.history['val_loss'], 'r', label='val loss')

        acc_ax.plot(hist.history['acc'], 'b', label='train acc')
        acc_ax.plot(hist.history['val_acc'], 'g', label='val acc')

        acc_ax.set_xlabel('epoch')# loss_ax.set_xlabel('epoch')
        # loss_ax.set_ylabel('loss')
        acc_ax.set_ylabel('accuray')

        # loss_ax.legend(loc='upper left')
        acc_ax.legend(loc='upper left')# acc_ax.legend(loc='lower left')

        self.loss.extend(hist.history['loss'])
        self.acc.extend(hist.history['acc'])
        self.val_loss.extend(hist.history['val_loss'])
        self.val_acc.extend(hist.history['val_acc'])
