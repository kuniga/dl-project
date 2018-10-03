import os
import shutil
import subprocess
from keras.models import Model
from keras.layers import GlobalAveragePooling2D
from keras.layers.core import Dense
from keras.optimizers import RMSprop
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.applications import InceptionV3
from keras.preprocessing.image import ImageDataGenerator


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


class MonkeyDataset():
    def __init__(self):
        self.image_shape = (128, 128, 3)
        self.num_classes = 10

    def extract_files(self):
        if os.path.isdir("training") and os.path.isdir("validation"):
            return
        subprocess.run("unzip -o dataset/10-monkey-species.zip".split())
        subprocess.run("unzip -o -q training.zip".split())
        subprocess.run("unzip -o -q validation.zip".split())
        os.remove("monkey_labels.txt")
        os.remove("training.zip")
        os.remove("validation.zip")

    def generator(self, directory):
        datagen = ImageDataGenerator(
            rescale=1.0 / 255,
            featurewise_center=False,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by std
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=False,  # apply ZCA whitening
            rotation_range=30,  # randomly rotate images in the range (0~180)
            width_shift_range=0.2,  # randomly shift images horizontally
            height_shift_range=0.2,  # randomly shift images vertically
            horizontal_flip=True,  # randomly flip images
            vertical_flip=False)  # randomly flip images

        return datagen.flow_from_directory(
            directory,
            target_size=self.image_shape[:2])


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
    epochs=20,
    validation_data=validation_generator)

# show result
score = model.evaluate_generator(validation_generator)
print("Test loss:", score[0])
print("Test accuracy:", score[1])
