from tensorflow.keras.datasets import fashion_mnist
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

(train_data, train_labels), (test_data, test_labels) = fashion_mnist.load_data()

# -----------------------------------------------
# Normalize data by deviding them to the maximum value

train_data = train_data / np.max(train_data)
test_data = test_data / np.max(test_data)

(train_data_row_numbers, train_data_columns_numbers) = train_data[1].shape

print("Type: ", type(train_data))

class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
               "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

# ------------------------------------------
# Create model

model = tf.keras.models.Sequential()
model.add(Flatten(input_shape=(train_data_row_numbers, train_data_columns_numbers)))
model.add(Dense(5, activation="relu"))
model.add(Dense(5, activation="relu"))
model.add(Dense(5, activation="relu"))
model.add(Dense(len(class_names), activation="softmax"))

# --------------------------------------------
# Compile the model

model.compile(
    loss='categorical_crossentropy',
    optimizer=tf.keras.optimizers.Adam(lr=.0001),
    metrics=["accuracy"])


# -------------------------------------
# Fit data

def scheduler(epoch, lr):
    if epoch < 3:
        return lr
    else:
        return lr * tf.math.exp(-0.1)

callbacks = [tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5),
             tf.keras.callbacks.LearningRateScheduler(scheduler),
             tf.keras.callbacks.ModelCheckpoint(filepath='./Multi-Class-Classification.h5')]

history = model.fit(train_data,
                    tf.one_hot(train_labels, depth=10), epochs=100,
                    validation_data=(test_data, tf.one_hot(test_labels, depth=10)),
                    callbacks=[callbacks])

# -------------------------------
# Visualize the loss and metric

# pd.DataFrame(non_norm_history.history).plot()
# plt.show()

# ------------------------------------
# Use visualization to find best leraning_rate

dt = pd.DataFrame(history.history)

dt = dt.loc[:, ["loss", "lr"]]
plt.scatter(dt["lr"], dt["loss"])
plt.xlabel("lr")
plt.ylabel("loss")
plt.show()
