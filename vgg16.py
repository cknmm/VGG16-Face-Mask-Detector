# import necessary package
import tensorflow as tf
import numpy as np
import pathlib
import datetime

from tensorflow.keras import datasets, layers, models, losses

# Raw Dataset Directory
data_dir = pathlib.Path("./Dataset/Train")
image_count = len(list(data_dir.glob('*/*.png')))
print(image_count)
# classnames in the dataset specified
CLASS_NAMES = np.array([item.name for item in data_dir.glob('*') if item.name != "LICENSE.txt" ])
print(CLASS_NAMES)
# print length of class names
output_class_units = len(CLASS_NAMES)
print(output_class_units)


model = models.Sequential()
#model.add(layers.experimental.preprocessing.Resizing(224, 224, interpolation="bilinear", input_shape=(128, 128, 128, 3)))
model.add(layers.Conv2D(64, 3, strides=1, padding='same', input_shape=(128, 128, 3)))
model.add(layers.Activation('relu'))
model.add(layers.Conv2D(64, 3, strides=1, padding='same'))
model.add(layers.Activation('relu'))
model.add(layers.MaxPooling2D(2, strides=2))
model.add(layers.Conv2D(128, 3, strides=1, padding='same'))
model.add(layers.Activation('relu'))
model.add(layers.Conv2D(128, 3, strides=1, padding='same'))
model.add(layers.Activation('relu'))
model.add(layers.MaxPooling2D(2, strides=2))
model.add(layers.Conv2D(256, 3, strides=1, padding='same'))
model.add(layers.Activation('relu'))
model.add(layers.Conv2D(256, 3, strides=1, padding='same'))
model.add(layers.Activation('relu'))
model.add(layers.Conv2D(256, 1, strides=1, padding='same'))
model.add(layers.Activation('relu'))
model.add(layers.MaxPooling2D(2, strides=2))
model.add(layers.Conv2D(512, 3, strides=1, padding='same'))
model.add(layers.Activation('relu'))
model.add(layers.Conv2D(512, 3, strides=1, padding='same'))
model.add(layers.Activation('relu'))
model.add(layers.Conv2D(512, 1, strides=1, padding='same'))
model.add(layers.Activation('relu'))
model.add(layers.MaxPooling2D(2, strides=2))
model.add(layers.Conv2D(512, 3, strides=1, padding='same'))
model.add(layers.Activation('relu'))
model.add(layers.Conv2D(512, 3, strides=1, padding='same'))
model.add(layers.Activation('relu'))
model.add(layers.Conv2D(512, 1, strides=1, padding='same'))
model.add(layers.Activation('relu'))
model.add(layers.MaxPooling2D(2, strides=2))
model.add(layers.Flatten())
model.add(layers.Dense(4096, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(4096, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(output_class_units, activation='softmax'))
model.summary()


# Shape of inputs to NN Model
BATCH_SIZE = 32             # Can be of size 2^n, but not restricted to. for the better utilization of memory
IMG_HEIGHT = 128            # input Shape required by the model
IMG_WIDTH = 128             # input Shape required by the model
STEPS_PER_EPOCH = np.ceil(image_count/BATCH_SIZE)

# Rescalingthe pixel values from 0~255 to 0~1 For RGB Channels of the image.
image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
# training_data for model training
train_data_gen = image_generator.flow_from_directory(directory=str(data_dir),
                                                     batch_size=BATCH_SIZE,
                                                     shuffle=True,
                                                     target_size=(IMG_HEIGHT, IMG_WIDTH), #Resizing the raw dataset
                                                     classes = list(CLASS_NAMES))
valid_generator = image_generator.flow_from_directory(
    directory=str(data_dir),
    batch_size=BATCH_SIZE,
    shuffle=True,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    classes = list(CLASS_NAMES)
)
#train_data_gen = np.array(train_data_gen)

# Specifying the optimizer, Loss function for optimization & Metrics to be displayed
model.compile(optimizer='sgd', loss="categorical_crossentropy", metrics=['accuracy'])

# Summarizing the model architecture and printing it out
model.summary()

epochs = 100
# Training the Model
history = model.fit(
      train_data_gen,
      validation_steps = valid_generator.n//valid_generator.batch_size,
      steps_per_epoch=STEPS_PER_EPOCH,
      epochs=epochs)

# Saving the model
model.save('VGG16 Face Mask Detection//')

score = model.evaluate_generator(valid_generator)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
