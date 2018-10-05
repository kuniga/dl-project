import os
import shutil
from keras.models import Model
from keras.layers import GlobalAveragePooling2D
from keras.layers.core import Dense
from keras.optimizers import RMSprop, SGD
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.applications import InceptionV3
from monkey_dataset import MonkeyDataset


def network(num_classes):
    base_model = InceptionV3(weights="imagenet", include_top=False)
    for layer in base_model.layers:
        layer.trainable = False
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, kernel_initializer="he_normal", activation="relu")(x)
    x = Dense(512, kernel_initializer="he_normal", activation="relu")(x)
    prediction = Dense(num_classes, activation="softmax")(x)
    model = Model(inputs=base_model.input, outputs=prediction)
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
        logdir = "logdir_monkey_pretrain_inceptionv3_with_aug"
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
model = network(dataset.num_classes)

# train the model
dataset.extract_files()
training_generator = dataset.generator('training')
validation_generator = dataset.generator('validation')

trainer = Trainer(
    model,
    loss="categorical_crossentropy",
    optimizer=RMSprop())

trainer.train(
    training_generator,
    epochs=8,
    validation_data=validation_generator)

for layer in model.layers[:249]:
    layer.trainable = False

for layer in model.layers[249:]:
    layer.trainable = True

trainer = Trainer(
    model,
    loss="categorical_crossentropy",
    optimizer=SGD(lr=0.001, momentum=0.9))

trainer.train(
    training_generator,
    epochs=8,
    validation_data=validation_generator)

# show result
score = model.evaluate_generator(validation_generator)
print("Test loss:", score[0])
print("Test accuracy:", score[1])
