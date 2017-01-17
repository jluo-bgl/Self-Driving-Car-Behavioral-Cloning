from model import nvida1
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.optimizers import Adam

class Trainer(object):
    def __init__(self, data_provider):
        self.data_provider = data_provider

    def generate_model(self):
        return nvida1()

    def fit(self):
        model = self.generate_model()
        model.summary()

        checkpointer = ModelCheckpoint(
            filepath="model.h5",
            monitor='val_mean_squared_error',
            verbose=1, save_best_only=True, mode='min'
        )

        # early_stop = EarlyStopping(monitor='val_mean_squared_error', min_delta=0.0001, patience=4, verbose=1, mode='min')

        adam = Adam(lr=0.001)
        model.compile(optimizer=adam, loss='mean_squared_error', metrics=['accuracy', 'mean_squared_error'])

        print('Starting training')

        model_json = model.to_json()
        with open("model.json", "w") as json_file:
            json_file.write(model_json)

        epochs = 5
        batch_size = 128
        model.fit(self.data_provider.images, self.data_provider.angles,
                  validation_split=0.2,
                  nb_epoch=epochs,
                  batch_size=batch_size,
                  shuffle=True,
                  callbacks=[checkpointer]
                  )
