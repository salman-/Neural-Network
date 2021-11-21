import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import Input
from tensorflow.keras.layers import Dense

from Dataset import Dataset

# The goal is to predict the insurance costs based on the other parameters

# ---------------------------------------
# Create TRAIN and TEST set
dataset = Dataset()
train_x = dataset.process_data(dataset.train_x)
train_y = dataset.train_y

test_x = dataset.process_data(dataset.test_x)
test_y = dataset.test_y
# ----------------------------------------
# Create Neural-Network

insurance_model = tf.keras.Sequential()
insurance_model.add(Input(shape=(11,)))
insurance_model.add(Dense(100, activation="relu"))
insurance_model.add(Dense(60, activation="relu"))
insurance_model.add(Dense(30, activation="relu"))
insurance_model.add(Dense(30, activation="relu"))
insurance_model.add(Dense(30, activation="relu"))
insurance_model.add(Dense(30, activation="relu"))
insurance_model.add(Dense(1, activation=None))

# -------------------------------------
# Compile the model

insurance_model.compile(
    loss=tf.keras.losses.mae,
    optimizer=tf.keras.optimizers.Adamax(learning_rate=0.001),
    metrics=["mae"]
)

# ------------------------------------
# Train the model
callbacks = [tf.keras.callbacks.EarlyStopping(monitor='loss', patience=100),
             tf.keras.callbacks.ModelCheckpoint(filepath='./insurance-predictor.h5')]
history = insurance_model.fit(train_x, train_y, epochs=10000, callbacks=callbacks)

# ------------------------------------
# Evaluate model on test dataset

print(" Evaluate model on test: ", insurance_model.evaluate(test_x, test_y))

pd.DataFrame(history.history).plot()
plt.ylabel("MAE")
plt.xlabel("Epochs")

plt.show()
