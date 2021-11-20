import pandas as pd
import plotly.express as px
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense, Input
from Dataset import Dataset
from sklearn.metrics import confusion_matrix

# Create sample circles
dt = Dataset()

# -------------------------------------------------
# Create Neural-Network

model = tf.keras.models.Sequential()
model.add(Input(shape=(2,)))
model.add(Dense(100, activation="relu"))
model.add(Dense(10, activation="relu"))
model.add(Dense(1, activation="sigmoid"))
# -------------------------------------------------
# Compile the network

model.compile(
    loss=tf.keras.losses.BinaryCrossentropy(),
    optimizer=tf.keras.optimizers.Adam(),
    metrics=["accuracy"]
)

# -------------------------------------------------
# Fit the model
callbacks = [tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10),
             tf.keras.callbacks.ModelCheckpoint(filepath='./binary-classification.h5')]
history = model.fit(dt.train, dt.label_train, epochs=100, callbacks=callbacks)

# -------------------------------------------------
# Evaluate model against test data
print("Evaluation result: ", model.evaluate(dt.test, dt.label_test))

# ---------------------------------------
# Predict and visualize the result

predicted_label = model.predict(dt.test)
predicted_label = (predicted_label < 0.5)

#--------------------------------------------------
# Visualize history of loss and metric

pd.DataFrame(history.history).plot()
plt.show()

# ----------------------------------------
# Visualize the test dataset against the train dataset

predicted_label = pd.DataFrame(predicted_label, columns=["Label"])

dt.test["Category"] = dt.create_category_label(predicted_label["Label"],
                                               "Test-External Circle", "Test-Internal Circle")
dt.train["Category"] = dt.create_category_label(dt.label_train,
                                                "Train-Internal Circle", "Train-External Circle")

df = pd.concat([dt.train, dt.test])
fig = px.scatter(df, x="Coordinate_X", y="Coordinate_Y", color="Category")
fig.show()

#----------------------------------------
# Get confusion matrix
print("Confusion Matrix: ")
print(confusion_matrix(dt.label_test,predicted_label))