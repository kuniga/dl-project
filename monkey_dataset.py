import os
import subprocess
from keras.preprocessing.image import ImageDataGenerator


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
