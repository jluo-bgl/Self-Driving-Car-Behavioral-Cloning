from model import nvida1
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.optimizers import Adam


class Trainer(object):
    def __init__(self, learning_rate, epoch):
        self.learning_rate = learning_rate
        self.epoch = epoch

    def generate_model(self):
        return nvida1()

    def fit(self, generator):
        model = self.generate_model()
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

        model.fit_generator(generator, samples_per_epoch=20000,
                            nb_epoch=self.epoch,
                            verbose=1,
                            callbacks=[checkpointer]
                            )
