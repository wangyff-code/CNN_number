import tensorflow as tf
from tensorflow import keras
from tensorflow_core.python.keras import Sequential
from tensorflow_core.python.keras.layers.core import Dense, Dropout, Flatten
from tensorflow_core.python.keras.layers import Conv2D, MaxPooling2D

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()


x_train = x_train.reshape(x_train.shape[0], 28, 28,1)
x_test = x_test.reshape(x_test.shape[0], 28, 28,1)



model = tf.keras.models.Sequential()

#-------------MOD------------------------
model.add(Conv2D(16, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=(28,28,1)))

model.add(MaxPooling2D(pool_size=(2, 2)))



model.add(Flatten())



model.add(Dense(10, activation='softmax'))
#-------------------------------------
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5)

model.evaluate(x_test,  y_test, verbose=2)

print("Saving model to disk \n")
mp = "iris_model.h5"
model.save(mp)

