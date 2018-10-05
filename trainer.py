import os
import shutil
from keras.callbacks import TensorBoard, ModelCheckpoint


class Trainer():
    def __init__(self, model, loss, optimizer, log_dir):
        self._target = model
        self._target.compile(
            loss=loss,
            optimizer=optimizer,
            metrics=["accuracy"]
        )
        self.verbose = 1
        self.log_dir = log_dir
        self.model_file_name = "model_file.hdf5"

    def train(self, training_data, epochs, validation_data):
        if os.path.exists(self.log_dir):
            shutil.rmtree(self.log_dir)  # remove previous execution
        os.mkdir(self.log_dir)

        model_path = os.path.join(self.log_dir, self.model_file_name)
        self._target.fit_generator(
            generator=training_data,
            epochs=epochs,
            validation_data=validation_data,
            callbacks=[
                TensorBoard(log_dir=self.log_dir),
                ModelCheckpoint(model_path, save_best_only=True)
            ],
            verbose=self.verbose,
            workers=4
        )
