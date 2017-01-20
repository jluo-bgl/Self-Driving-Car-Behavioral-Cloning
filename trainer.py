from model import nvida1
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.optimizers import Adam


class Trainer(object):
    def __init__(self, learning_rate, epoch, dropout=0.5, multi_process=False, number_of_worker=4):
        self.learning_rate = learning_rate
        self.epoch = epoch
        self.multi_process = multi_process
        self.number_of_worker = number_of_worker
        self.dropout = dropout

    def generate_model(self, input_shape, dropout):
        return nvida1(input_shape, dropout)

    def fit(self, generator, input_shape):
        model = self.generate_model(input_shape, self.dropout)
        model.summary()

        checkpointer = ModelCheckpoint(
            filepath="model.h5",
            save_weights_only=True,
            verbose=1
        )

        # early_stop = EarlyStopping(monitor='val_mean_squared_error', min_delta=0.0001, patience=4, verbose=1, mode='min')

        adam = Adam(lr=self.learning_rate)
        model.compile(optimizer=adam, loss='mean_squared_error', metrics=['accuracy', 'mean_squared_error'])

        print('Starting training')

        model_json = model.to_json()
        with open("model.json", "w") as json_file:
            json_file.write(model_json)

        model.fit_generator(generator, samples_per_epoch=16384,
                            nb_epoch=self.epoch,
                            verbose=1,
                            nb_worker=self.number_of_worker,
                            max_q_size=20,
                            pickle_safe=self.multi_process,
                            callbacks=[checkpointer]
                            )
