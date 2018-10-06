import os
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten, Dropout
from keras.layers.core import Dense
from keras.optimizers import RMSprop
from monkey_dataset import MonkeyDataset
from trainer import Trainer

TPU_WORKER = 'grpc://' + os.environ['COLAB_TPU_ADDR']


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


dataset = MonkeyDataset()

# make model
model = network(dataset.image_shape, dataset.num_classes)

# train the model
training_generator = dataset.generator('training')
validation_generator = dataset.generator('validation')

trainer = Trainer(
    model,
    loss="categorical_crossentropy",
    optimizer=RMSprop(),
    log_dir="logdir_monkey_deep_with_aug")

trainer.train(
    training_generator,
    epochs=200,
    validation_data=validation_generator)

# show result
score = model.evaluate_generator(validation_generator)
print("Test loss:", score[0])
print("Test accuracy:", score[1])
