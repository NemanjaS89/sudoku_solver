import keras
import tensorflow as tf
import matplotlib.pyplot as plt
import os


mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
"""""
plt.imshow(x_train[0], cmap = plt.cm.binary)
plt.show()
"""""
x_train = tf.keras.utils.normalize(x_train, axis = 1)
x_test = tf.keras.utils.normalize(x_test, axis = 1)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation = tf.nn.relu))
model.add(tf.keras.layers.Dense(128, activation = tf.nn.relu))
model.add(tf.keras.layers.Dense(10, activation = tf.nn.softmax))

model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy',
              metrics = ['accuracy'])
model.fit(x_train, y_train, epochs = 3)
val_loss, val_acc = model.evaluate(x_test, y_test)
print(val_loss, val_acc)

model_json = model.to_json()
with open ('model_json', 'w') as json_file:
    json_file.write(model_json)

model.save_weights('model.h5')