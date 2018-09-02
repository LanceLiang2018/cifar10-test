import keras
from keras.layers.core import *
from keras.layers.convolutional import *
from keras.models import Sequential, load_model
from random import randint
from keras.utils import plot_model


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


test_name = 'cifar1'
count = 10000
epoch = 5
rand = randint(0, 10000 - count)
count = rand + count

batch = unpickle('data_batch_1')
test_batch = unpickle('test_batch')
print(batch.keys())
print(batch[b'data'])
meta = unpickle('batches.meta')
print(meta)

# (x_train, y_train), (x_test, y_test) =
x_train, x_test = batch[b'data'], test_batch[b'data']
y_train, y_test = batch[b'labels'], test_batch[b'labels']
x_train, x_test = x_train / 255.0, x_test / 255.0

x_train = x_train.reshape(x_train.shape[0], 32, 32, 3)
x_test = x_test.reshape(x_test.shape[0], 32, 32, 3)

try:
    model = load_model('%s.hdf5' % test_name)
except Exception as e:
    model = Sequential()
    model.add(Conv2D(16, kernel_size=(3, 3), strides=(1, 1),
                     activation='relu',
                     input_shape=(32, 32, 3),
                     name='C1'))
    model.add(MaxPooling2D(pool_size=(4, 4), strides=(2, 2), name='S1'))
    model.add(Conv2D(8, (3, 3), activation='relu', name='C2'))
    model.add(MaxPooling2D(pool_size=(2, 2), name='S2'))
    model.add(Conv2D(6, (3, 3), activation='relu', name='C3'))
    model.add(MaxPooling2D(pool_size=(2, 2), name='S3'))
    model.add(Flatten())
    model.add(Dense(320, activation='relu'))
    model.add(Dense(84, activation='relu'))
    model.add(Dense(10, activation='softmax'))

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

model.fit(x_train[rand:count], y_train[rand:count], epochs=epoch)

model.save('%s.hdf5' % test_name)

print(model.evaluate(x_test, y_test))
print(model.metrics_names)
