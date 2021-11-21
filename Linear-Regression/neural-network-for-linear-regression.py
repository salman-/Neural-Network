import tensorflow as tf
from tensorflow.keras import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import load_model

import numpy as np
import matplotlib.pyplot as plt

TEST_TRAIN_PORTION = 0.2
SAMPLE_SIZE = 2500

#------------------------------------------------------
# Create a sample for linear-regression

x = np.arange(SAMPLE_SIZE)
m = 5
b = 4
y = m * x + b

#-------------------------------------------------------
# Split sample to TEST and TRAIN

splitting_index = int(x.size * (1 - TEST_TRAIN_PORTION))

train_x = x[:splitting_index]
train_y = y[:splitting_index]

test_x = x[splitting_index:]
test_y = y[splitting_index:]

# -----------------------------------------------------
# Create a Sequential Nerual Network
model = tf.keras.Sequential()
model.add(Input(shape=(1,), name="input_layer"))
model.add(Dense(10, activation="relu"))
model.add(Dense(1, activation=None, name="output_layer"))

# -----------------------------------------------------
# Compile the model
model.compile(loss=tf.keras.losses.mae,
              optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
              metrics=["mae"])

# -----------------------------------------------------
# Train the model
model.fit(train_x, train_y, epochs=250)

#-----------------------------------------------------
# Get the summary of the model and visualize it

#model.summary()
plot_model(model=model, show_shapes=True)

# -----------------------------------------------------
# Test the model

predicted_value = model.predict(test_x)

# ------------------------------------------------------
# Evaluation


#-------------------------------------------------------
#Save the model with h5 format and reload it

model.save("./Linear-regression.h5")

model_1 = load_model("Linear-regression.h5")
model_1.summary()

# -----------------------------------------------------
# Visualization of test and train data
plt.plot(train_x, train_y, color="b", label="Train Data", alpha=0.7)
plt.plot(test_x, test_y, color="r", label="Test Data" ,alpha=0.7)
plt.plot(test_x, predicted_value, color="g", label="Predicted Test Data", linestyle='--', alpha=0.7)
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.show()


