#EXPLAINERS:

#CNN MEANS THAT EACH NEURON ON THE INPUT IS REDUCED DOWN TO PROVIDE THE OUTPUT
#The following hyperparams are relevant: InputSize, Padding, Kernel Size, Stride
#Input- the amount of inputs x,y. 7X7 etc
#Padding computationally expensive but if data needs to be extracted from the edges, adding padding is a good idea....
#... otherwise the filter has less passes over the relevant info on the edges
#Kernel size (output size) - the scale in which each input correlates to the outpus. for example a 7X7 image with a Kernel size of 5 turns into a 3X3 output
#Stride - are the steps the filter passes over the picture to get the information. Higher stride reduces computation, lower stride is more...
#...thorough and extracts more information from the image


import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import tensorflow_datasets as tfds
from tensorflow.keras.models import Model
from tensorflow.keras.losses import BinaryCrossentropy
from keras.layers import MaxPool2D, Conv2D, Dense, Flatten, InputLayer, BatchNormalization, Input,  Layer
from tensorflow.keras.optimizers import Adam


#loads tensorflow provided dataset
dataset, dataset_info = tfds.load('malaria', with_info=True, as_supervised=True, shuffle_files=True, split=['train'])


# helper function to prepare data
def splitDatasets(dataset, TRAIN_RATIO, VAL_RATIO, TEST_RATIO):
  DATASET_SIZE = len(dataset)

  #.take takes the first n elements as specified as argument
  train_dataset = dataset.take(int(TRAIN_RATIO * DATASET_SIZE))
  #.skip skips the first n elements as specified as argument
  validation_and_training_dataset = dataset.skip(int(TRAIN_RATIO * DATASET_SIZE))
  val_dataset = validation_and_training_dataset.take(int(VAL_RATIO * DATASET_SIZE))
 
  test_dataset = validation_and_training_dataset.skip(int(VAL_RATIO * DATASET_SIZE))
  
  return train_dataset, val_dataset, test_dataset



# Prepare the data
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

train_dataset, val_dataset, test_dataset = splitDatasets(dataset[0],TRAIN_RATIO, VAL_RATIO, TEST_RATIO)


# visualize the data

#check the dataset_info (or similar) for the labels to destructure!!!

# tracks i for each iteration, destructures image, label from the dataset. 
# enumerate assigns an index to the data starting with 0
# take 16, visualizes only the first 16 elements from the dataset to save time
for i, (image, label) in enumerate(train_dataset.take(16)):
  #5, 5 is the image arrangement, i + 1 is the image position for each iteration
  ax = plt.subplot(5,5, i+1)
  plt.imshow(image)
  plt.title(dataset_info.features['label'].int2str(label))


  # Data Pre-Processing

#explainer Image shape is (IM_SIZE, IM_SIZE, 3) - the 3 refers to the 3 complimentary colors, color channels

# Images need to be resized to a fix width and height and rescaled
# Images have values between 0-255. They are not standardized however, the image is normalized

IM_SIZE = 224
# callback to resize the image
def resize_and_rescale(image, label):
  #tf.image.resize(image, (IM_SIZE, IM_SIZE)) = resizing
  #/255 = rescaling
  return tf.image.resize(image, (IM_SIZE, IM_SIZE))/255, label


# mutates the dataset with the resizing method
train_dataset = train_dataset.map(resize_and_rescale)
val_dataset = val_dataset.map(resize_and_rescale)


#confirm that the image has the correct proportions as mutated above
for image, label in train_dataset.take(1):
  print(image)


  #Pre-Processing part 2

#shuffle the dataset
BATCH_SIZE = 32
train_dataset = train_dataset.shuffle(buffer_size = 8, reshuffle_each_iteration=True).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
val_dataset = val_dataset.shuffle(buffer_size = 8, reshuffle_each_iteration=True).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)



# Model creation
# "3" in input layer refers to the channels of the image - RGB in this case

normalizer = tf.keras.layers.Conv2D(filters = 6, kernel_size = 5, strides=1, padding="valid", activation="sigmoid")
#LeNet CNN
# SEQUENTIAL API
model = tf.keras.Sequential([
    InputLayer(input_shape=(IM_SIZE, IM_SIZE, 3)),
    #Convolution 1 + Pool
    tf.keras.layers.Conv2D(filters = 6, kernel_size = 5, strides=1, padding="valid", activation="sigmoid"),
    MaxPool2D(pool_size=2, strides=2),
    #Convolution 2 + Pool
    tf.keras.layers.Conv2D(filters = 16, kernel_size = 5, strides=1, padding="valid", activation="sigmoid"),
    MaxPool2D(pool_size=2, strides=2),

    Flatten(),

    Dense(100, activation='sigmoid'),
    Dense(10, activation='sigmoid'),
    # 0 for infected and 1 for uninfected. We have one output
    Dense(1, activation='sigmoid'),
])
model.summary()


#Error sanctioning explanation below not needed is done during compilation
bce = tf.keras.losses.BinaryCrossentropy(from_logits=False) #False(default) applies if Sigmoid is used which has a 0-1 

# Model Compilation

model.compile(optimizer= Adam(learning_rate = 0.1),
              loss = BinaryCrossentropy(),
              metrics = [TruePositives(name="TP"), FalsePositives(name="FP"), TrueNegatives(name="TN"), FalseNegatives(name="FN"), BinaryAccuracy(name="accuracy"), Precision(name="precision"), Recall(name="recall"), AUC(name="auc")]
              )

history = lenetClass.fit(train_dataset, validation_data=val_dataset ,epochs = 5, verbose=1)


#==========================================
#FUNCTIONAL BASED


#LeNet CNN
func_input = Input(shape= (IM_SIZE, IM_SIZE, 3), name="Input Image")

#THESE convolution LAYERS EXTRACT THE FEATURES
x = tf.keras.layers.Conv2D(filters = 12, kernel_size = 3, strides=1, padding="valid", activation="relu")(func_input)
x =   BatchNormalization()(x)
x =    MaxPool2D(pool_size=2, strides=2)(x)
    
x =    tf.keras.layers.Conv2D(filters = 12, kernel_size = 3, strides=1, padding="valid", activation="relu")(x)
x =    BatchNormalization()(x)
output =    MaxPool2D(pool_size=2, strides=2)(x)
feature_extractor_model = Model(func_input, output, name="Lenet_Model")


#THESE LAYERS CLASSIFY IF THE IMAGE IS PARASITIC
x = feature_extractor_model(func_input)
x =   Flatten()(x)
x =   Dense(220, activation="relu")(x)
x =   BatchNormalization()(x)
x =   Dense(75, activation="relu")(x)
x =   BatchNormalization()(x)

# 0 for infected and 1 for uninfected. We have one output
func_output =  Dense(1, activation='sigmoid')(x)

model_func = Model(func_input, func_output, name="Lenet_Model")
model_func.summary()


model_func.compile(optimizer= Adam(learning_rate = 0.1),
              loss = BinaryCrossentropy(),
              metrics = [TruePositives(name="TP"), FalsePositives(name="FP"), TrueNegatives(name="TN"), FalseNegatives(name="FN"), BinaryAccuracy(name="accuracy"), Precision(name="precision"), Recall(name="recall"), AUC(name="auc")]
              )

history = lenetClass.fit(train_dataset, validation_data=val_dataset ,epochs = 5, verbose=1)


#=========================================

#OOP class based

#SUBCLASSING - OOP CLASSED BASED
#below inherits from Layer (keras)
class FeatureExtractor(Layer):
  def __init__(self, filters, kernel_size, strides, padding, activation, pool_size):
    #inherits Layer through FeatureExdtractor and initializes the constructor
    super(FeatureExtractor, self).__init__()
    
    self.convulution_1 = tf.keras.layers.Conv2D(filters = filters, kernel_size = kernel_size, strides=strides, padding=padding, activation=activation)
    self.batch_1 = BatchNormalization()
    self.pool_1 = MaxPool2D(pool_size=pool_size, strides=strides * 2)

    self.convulution_2 = tf.keras.layers.Conv2D(filters = filters * 2, kernel_size = kernel_size, strides=strides, padding=padding, activation=activation)
    self.batch_2 = BatchNormalization()
    self.pool_2 = MaxPool2D(pool_size=pool_size, strides=strides * 2)

    def call(self, x, training ):
      self.convolution_1(x)
      self.batch_1(x)
      self.pool_1(x)
      self.convolution_2(x)
      self.batch_2(x)
      self.pool_2(x)

      return x

class LenetModel(Model):
  def __init__(self):
    super(LenetModel, self).__init__()
    self.feature_extractor = FeatureExtractor(8, 3, 1, "valid", "relu", 2)
    self.flatten = Flatten()

    self.dense_1 = Dense(220, activation="relu")
    self.batch_1 = BatchNormalization()
    self.dense_2 = Dense(75, activation="relu")
    self.batch_2 = BatchNormalization()
    self.dense_3 = Dense(1, activation="sigmoid")

  def call(self, x, training):
    x = self.feature_extractor(x)
    x = self.flatten(x)
    x = self.dense_1(x)
    x = self.batch_1(x)
    x = self.dense_2(x)
    x = self.batch_2(x)
    x = self.dense_3(x)
    return x

lenetClass = LenetModel()
lenetClass(tf.zeros([1, 224, 224, 3]))


# DEFINING CUSTOM DENSE LAYER

class CustomDense(Layer):
  def __init__(self, output_units):
    super(CustomDense, self).__init__()
    self.output_units = output_units

  def build(self, input_features_shape):
    #add_weight inherited from Layer,  input_features_shape[-1] == number of rows in matrix
    self.w = self.add_weight(shape = (input_features_shape[-1], self.output_units), initializer="random_normal", trainable = True)
    self.b = self.add_weight(shape = (self.output_units), initializer="random_normal", trainable="Ture")

  def call(self, input_features):
    #matrix multiplyer
    if(activation == "relu"):
      return tf.nn.sigmoid(tf.matmul(input_features, self.w) + self.bi)
    elif (activation == "sigmoid"):
      return tf.math.sigmoid(tf.matmul(input_features, self.w) + self.b)
    else: return tf.matmul(input_features, self.w) + self.b