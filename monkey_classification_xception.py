from keras.models import Model
from keras.layers import GlobalAveragePooling2D
from keras.layers.core import Dense
from keras.optimizers import RMSprop, SGD
from keras.applications import xception
from monkey_dataset import MonkeyDataset
from trainer import Trainer


def network(num_classes):
    base_model = xception.Xception(weights="imagenet", include_top=False)
    for layer in base_model.layers:
        layer.trainable = False
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, kernel_initializer="he_normal", activation="relu")(x)
    x = Dense(512, kernel_initializer="he_normal", activation="relu")(x)
    prediction = Dense(num_classes, activation="softmax")(x)
    model = Model(inputs=base_model.input, outputs=prediction)
    return model


dataset = MonkeyDataset()

# make model
model = network(dataset.num_classes)

# train the model
training_generator = dataset.generator('training')
validation_generator = dataset.generator('validation')

trainer = Trainer(
    model,
    loss="categorical_crossentropy",
    optimizer=RMSprop(),
    log_dir="logdir_monkey_pretrain_xception_with_aug")

trainer.train(
    training_generator,
    epochs=8,
    validation_data=validation_generator)

for layer in model.layers[:105]:
    layer.trainable = False

for layer in model.layers[105:]:
    layer.trainable = True

trainer = Trainer(
    model,
    loss="categorical_crossentropy",
    optimizer=SGD(lr=0.001, momentum=0.9),
    log_dir="logdir_monkey_pretrain_xception_with_aug")

trainer.train(
    training_generator,
    epochs=50,
    validation_data=validation_generator)

# show result
score = model.evaluate_generator(validation_generator)
print("Test loss:", score[0])
print("Test accuracy:", score[1])
