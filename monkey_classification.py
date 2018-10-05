import os
import shutil
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten, Dropout
from keras.layers.core import Dense
from keras.optimizers import RMSprop
from keras.callbacks import TensorBoard, ModelCheckpoint
from monkey_dataset import MonkeyDataset


def network(input_shape, num_classes):
    model = Sequential()

    # extract image features by convolution and max pooling layers
    model.add(Conv2D(
        32, kernel_size=3, padding="same",
        input_shape=input_shape, activation="relu"
        ))
    model.add(Conv2D(32, kernel_size=3, activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(64, kernel_size=3, padding="same", activation="relu"))
    model.add(Conv2D(64, kernel_size=3, activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    # classify the class by fully-connected layers
    model.add(Flatten())
    model.add(Dense(512, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes))
    model.add(Activation("softmax"))
    return model


class Trainer():
    def __init__(self, model, loss, optimizer):
        self._target = model
        self._target.compile(
            loss=loss,
            optimizer=optimizer,
            metrics=["accuracy"]
        )
        self.verbose = 1
        logdir = "logdir_monkey_deep_with_aug"
        self.log_dir = logdir
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


dataset = MonkeyDataset()

# make model
model = network(dataset.image_shape, dataset.num_classes)

# train the model
training_generator = dataset.generator('training')
validation_generator = dataset.generator('validation')

trainer = Trainer(
    model,
    loss="categorical_crossentropy",
    optimizer=RMSprop())

trainer.train(
    training_generator,
    epochs=200,
    validation_data=validation_generator)

# show result
score = model.evaluate_generator(validation_generator)
print("Test loss:", score[0])
print("Test accuracy:", score[1])
