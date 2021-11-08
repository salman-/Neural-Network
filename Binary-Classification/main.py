from sklearn.datasets import make_circles
import pandas as pd
import matplotlib.pyplot as plt

from tensorflow.keras.layers import Dense, Input
import tensorflow as tf

# Create sample circles
tf.random.set_seed(42)

n_sample = 1000
X, y = make_circles(n_sample, noise=0.03, random_state=42)


dt = pd.DataFrame({"x": X[:, 0], "y": X[:, 1], "Label": y[:]})

plt.scatter(x=dt["x"], y=dt["y"], c=dt["Label"])
plt.show()

# -------------------------------------------------
# Create Neural-Network

model = tf.keras.models.Sequential()
model.add(Input(shape=(2,)))
model.add(Dense(100,activation="relu"))
model.add(Dense(10,activation="relu"))
model.add(Dense(1,activation=None))
# -------------------------------------------------
# Compile the network

model.compile(
    loss=tf.keras.losses.BinaryCrossentropy(),
    optimizer=tf.keras.optimizers.Adam(),
    metrics=["accuracy"]
)

#-------------------------------------------------
# Fit the model

model.fit(X,y, epochs=100)
