import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(-1, 28*28).astype('float32') / 255.0
x_test = x_test.reshape(-1, 28*28).astype('float32') / 255.0

# Sequential API (very convenient, not very flexible)
# only allows you to map one input to one output
# layers.Dense means fully connected layer

model = keras.Sequential(
    [
     keras.Input(shape=(28*28)),
     layers.Dense(512, activation='relu'),
     layers.Dense(256, activation='relu'),
     layers.Dense(10, activation='softmax'),
     ]
)

#model = keras.Sequential()
#model.add(keras.Input(shape=(28*28))
#model.add(layers.Dense(512, activation='relu'))
#print(model.summary())
#model.add(layers.Dense(256, activation='relu'))
#model.add(layer.Dense(10)

#print(model.summary())
#import sys
#sys.exit()

# Functional API (a bit more flexible, can handle multiple inputs and multiple outputs)
#inputs = keras.Input(shape=(28*28))
#x = layers.Dense(512, activation='relu', name='first-layer')(inputs)
#x = layers.Dense(256, activation='relu', name='second-layer')(x)
#outputs = layers.Dense(10, activation='softmax')(x)
#model = keras.Model(inputs=inputs, outputs=outputs)

#get output of second layer
#model = keras.Model(inputs=inputs, outputs=[model.layers[-2].output])
#model = keras.Model(inputs=inputs, outputs=[model.get_layer('second-layer').output])

#model = keras.Model(inputs=inputs, outputs=[layer.output for layer in model.layers])
#features = model.predict(x_train)
#for feature in features:
#    print(feature.shape)

#print(model.summary())
#import sys
#sys.exit()

model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    optimizer=keras.optimizers.Adam(lr=0.001),
    metrics=["accuracy"]
)

model.fit(x_train, y_train, batch_size=32, epochs=5, verbose=2)
model.evaluate(x_test, y_test, batch_size=32, verbose=2)