import numpy as np
from keras.preprocessing.image import load_img, img_to_array
from keras.models import load_model
from tensorflow.keras.models import load_model

longitudee, high = 150, 150
model = "./modelo/modelo.h5"
weights_model = "./modelo/pesos.h5"
cnn = load_model(model)
cnn.load_weights(weights_model)


def predict(file):
    x = load_img(file, target_size=(longitudee, high))
    x = img_to_array(x)
    x = np.expand_dims(x, axis=0)
    array = cnn.predict(x)
    result = array[0]
    answer = np.argmax(result)
    if answer == 0:
        print("pred: a")
    elif answer == 1:
        print("pred: b")
    elif answer == 2:
        print("pred: c")

    return answer


predict("pruebab.jpeg")
