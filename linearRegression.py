#Linear Regression


import tensorflow as tf
import pandas as pd
import numpy as np
import seaborn as sns
from tensorflow.keras.layers import Normalization, Dense, InputLayer
from tensorflow.keras.losses import MeanSquaredError, Huber, MeanAbsoluteError
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import RootMeanSquaredError
import matplotlib.pyplot as plt


# Reads the CSV
data = pd.read_csv("train.csv", ",")

# Plots the data via seabird for first inspection
plotted_data = sns.pairplot(data[["years","km","rating","condition","economy","top speed","hp","torque","current price"]], diag_kind="kde")

# Creates the tensor flow data structure
tensor_data = tf.constant(data)

# Shuffles the data
tensor_data = tf.random.shuffle(tensor_data)


#----------Data preparation-----------------

#define inputs (X - features)
X = tensor_data[: , 3:-1]


#define output (Y - price)
Y = tensor_data[:, -1]
# expands dimensions if shape is not the same
Y = tf.expand_dims(Y, axis=-1)

# Defines the rations to split the set into training, validation, test
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1
DATASET_SIZE = len(X)

# training set is beginning of data to the end of dataset * training ratio (800 items)
X_train = X[: int(DATASET_SIZE * TRAIN_RATIO)]
Y_train = Y[: int(DATASET_SIZE * TRAIN_RATIO)]

# valitation set creation from the end of training data to the end of training+validation
X_val = X[int(DATASET_SIZE * TRAIN_RATIO) : int(DATASET_SIZE * (TRAIN_RATIO+VAL_RATIO))]
Y_val = Y[int(DATASET_SIZE * TRAIN_RATIO) : int(DATASET_SIZE * (TRAIN_RATIO+VAL_RATIO))]

# test set from en of training+validation to end
X_test = X[int(DATASET_SIZE * (TRAIN_RATIO+VAL_RATIO)):]
Y_test = Y[int(DATASET_SIZE * (TRAIN_RATIO+VAL_RATIO)):]
print(X)
#MAKE SURE NOT DATA LEAKS BETWEEN SETS
#ONLY NORMALIZE THE TRAINING SET!!!


# Speeds up training process TUNING
train_dataset = tf.data.Dataset.from_tensor_slices((X_train, Y_train))
train_dataset = train_dataset.shuffle(buffer_size=8, reshuffle_each_iteration = True).batch(32).prefetch(tf.data.AUTOTUNE)

val_dataset = tf.data.Dataset.from_tensor_slices((X_val, Y_val))
val_dataset = train_dataset.shuffle(buffer_size=8, reshuffle_each_iteration = True).batch(32).prefetch(tf.data.AUTOTUNE)

val_test = tf.data.Dataset.from_tensor_slices((X_test, Y_test))
val_test = train_dataset.shuffle(buffer_size=8, reshuffle_each_iteration = True).batch(32).prefetch(tf.data.AUTOTUNE)

#Normalize the inputs
normalizer = Normalization()
normalizer.adapt(X_train) — only normalize training data


#-----------Modeling------------


# Instantiates the Model
model = tf.keras.Sequential([
    InputLayer(input_shape=(8,)), - defines the amount of features that is taken in “mileage”, “hp” etc. 
    normalizer,
    # If your model doesn’t perform as it should, adjust the DenseLayers and/or add additional neurons to each
    Dense(128, activation="relu"),
    Dense(128, activation="relu"),
    Dense(128, activation="relu"),
    Dense(1)
])

# starts the training process of the model
model.compile(optimizer=Adam(learning_rate=1), loss=MeanAbsoluteError(), metrics=RootMeanSquaredError())
history = model.fit(train_dataset, validation_data=val_dataset, epochs=100, verbose=0)


#----------Plotting--------------

# Plots the loss (should go down if training is effective) and the loss of the validation data
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.ylabel("Loss")
plt.xlabel("Epoch")
plt.legend(["Training", "Validation"])
plt.show()


# Evaluates the loss on the test set as a whole - all data
model.evaluate(X_test, Y_test)

# Uses the test data to make an actual prediction
y_pred = list(model.predict(X_test)[: , 0])

# Outputs the actual price from the sheet
y_true = list(Y_test[: ,0].numpy())


# Boilerplate code that plots a graph of test data results vs the actual values
ind = np.arange(100)
plt.figure(figsize=(40, 20))

width = 0.4

plt.bar(ind, y_pred, width, label="Predicted Car Price")
plt.bar(ind + width, y_true, width, label="Actual car price")

plt.xlabel("Actual vs Predicted Prices");
plt.ylabel("Car Prices")

plt.show()




