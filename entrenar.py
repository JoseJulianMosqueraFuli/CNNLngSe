import sys
import os
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras import optimizers
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dropout, Flatten, Dense, Activation
from tensorflow.python.keras.layers import Convolution2D, MaxPooling2D
from tensorflow.python.keras import backend as K


K.clear_session()

training_data = "./data/entrenamiento"
validation_data = "./data/validacion"

# Parameters

epoch = 20
longitude, high = 150, 150
batch_size = 32
steps = 1000
validation_steps = 300
conv1_filter = 32
conv2_filter = 64
filter1_size = (3, 3)
filter2_size = (2, 2)
pool_size = (2, 2)
classes = 3
lr = 0.0004


##Preparamos nuestras imagenes

training_data_generation = ImageDataGenerator(
    rescale=1.0 / 255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True
)

test_data_generation = ImageDataGenerator(rescale=1.0 / 255)

training_validator = training_data_generation.flow_from_directory(
    training_data,
    target_size=(high, longitude),
    batch_size=batch_size,
    class_mode="categorical",
)

generator_validate = test_data_generation.flow_from_directory(
    validation_data,
    target_size=(high, longitude),
    batch_size=batch_size,
    class_mode="categorical",
)

cnn = Sequential()
cnn.add(
    Convolution2D(
        conv1_filter,
        filter1_size,
        padding="same",
        input_shape=(longitude, high, 3),
        activation="relu",
    )
)
cnn.add(MaxPooling2D(pool_size=pool_size))

cnn.add(Convolution2D(conv2_filter, filter2_size, padding="same"))
cnn.add(MaxPooling2D(pool_size=pool_size))

cnn.add(Flatten())
cnn.add(Dense(256, activation="relu"))
cnn.add(Dropout(0.5))
cnn.add(Dense(classes, activation="softmax"))

cnn.compile(
    loss="categorical_crossentropy",
    optimizer=optimizers.Adam(lr=lr),
    metrics=["accuracy"],
)


cnn.fit_generator(
    training_validator,
    steps_per_epoch=steps,
    epochs=epoch,
    validation_data=generator_validate,
    validation_steps=validation_steps,
)

target_dir = "./modelo/"
if not os.path.exists(target_dir):
    os.mkdir(target_dir)
cnn.save("./modelo/modelo.h5")
cnn.save_weights("./modelo/pesos.h5")
