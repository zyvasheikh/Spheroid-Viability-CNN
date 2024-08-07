import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import warnings

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.optimizers import Adam
from tensorflow.keras.applications import VGG16

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization, Activation, Multiply, Reshape, Dropout

RGBtest_path = '/home/zyvasheikh/Desktop/work/images HITL/spheroid HITL 4C/RGBtest'

img_height = 224 #1536 #can make smaller to 224
img_width = 224 #2048 #224
batch_size = 10 #?
datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    vertical_flip = True,
    fill_mode='nearest',
    data_format = 'channels_last',
    validation_split = 0.1,
)

RGBtest_batches = datagen.flow_from_directory(directory = RGBtest_path, target_size = (img_height, img_width), classes = ['0-20%','20-40%','40-70%','70-100%'], batch_size = 10, shuffle = False, color_mode="rgb")

from keras.models import load_model

# Load the model
model = load_model('VGG16_model')

predictions = model.predict(RGBtest_batches)

y_pred = np.argmax(predictions,axis=1)
y_true = RGBtest_batches.classes


# test_loss, test_acc = model.evaluate(RGBtest_batches, verbose=2)
# print('/nTest accuracy:', test_acc)

#LEARNING CURVE
# Plot training & validation accuracy values
# plt.figure(figsize=(10, 5))
# accuracy = np.mean(true_labels == predicted_classes)
# print(accuracy)
# plt.plot(accuracy)
# plt.title('Test Accuracy')
# plt.xlabel('Sample')
# plt.ylabel('Accuracy')
# plt.savefig('/home/zyvasheikh/Desktop/work/PLOTS/VGG16 test accuracy 1.png')


# # Plot training & validation loss values
# plt.figure(figsize=(10, 5))
# plt.plot(history.history['loss'])
# loss = model.evaluate(RGBtest_batches)
# print(loss)
# plt.plot(loss)
# plt.title('Test Loss')
# plt.xlabel('Sample')
# plt.ylabel('Loss')
# plt.savefig('/home/zyvasheikh/Desktop/work/PLOTS/VGG16 test loss 1.png')

# #CONFUSION MATRIX
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn import metrics
import seaborn as sns

true_classes = RGBtest_batches.classes
class_labels = list(RGBtest_batches.class_indices.keys())

# Print the classification report
report = classification_report(y_true, y_pred, target_names=class_labels)
print(report)

# Print the confusion matrix
conf_matrix = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.savefig('/home/zyvasheikh/Desktop/work/PLOTS/VGG16 test confusion matrix 3.png')
