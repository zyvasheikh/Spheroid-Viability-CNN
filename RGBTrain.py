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


RGBtrain_path = '/home/zyvasheikh/Desktop/work/images HITL/spheroid HITL 4C/RGBtrain'
RGBtest_path = '/home/zyvasheikh/Desktop/work/images HITL/spheroid HITL 4C/RGBtest' #20-40% missing 2 images
RGBvalid_path = '/home/zyvasheikh/Desktop/work/images HITL/spheroid HITL 4C/RGBvalid'

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

RGBtrain_batches = datagen.flow_from_directory(directory = RGBtrain_path, target_size = (img_height, img_width), classes = ['0-20%','20-40%','40-70%','70-100%'], batch_size = 10, color_mode="rgb")
RGBvalid_batches = datagen.flow_from_directory(directory = RGBvalid_path, target_size = (img_height, img_width), classes = ['0-20%','20-40%','40-70%','70-100%'], batch_size = 10, shuffle = False, color_mode="rgb")
RGBtest_batches = datagen.flow_from_directory(directory = RGBtest_path, target_size = (img_height, img_width), classes = ['0-20%','20-40%','40-70%','70-100%'], batch_size = 10, shuffle = False, color_mode="rgb")

base_model = VGG16(weights='imagenet', include_top=False, input_shape=(img_width, img_height, 3))

# Freeze the base model
for layer in base_model.layers:
    layer.trainable = False

# Custom layers on top of VGG16
model = Sequential()
model.add(base_model)
model.add(BatchNormalization())
model.add(Flatten())
model.add(Dense(256,activation="relu"))
#model.add(Dropout(0.5))
model.add(Dense(4, activation='softmax'))

model.summary()

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.00001), loss='categorical_crossentropy', metrics=['accuracy'])

 #early_stopping = keras.callbacks.EarlyStopping(
#     monitor='val_loss',  # Metric to monitor for improvement (e.g., validation loss)
#     patience=5,           # Number of epochs with no improvement before stopping
#     restore_best_weights=True  # Restores the model weights to the best achieved during training
# )
# callbacks=[early_stopping]

# Train the model
history = model.fit(x=RGBtrain_batches, validation_data = RGBvalid_batches, epochs=100, verbose = 2)

#Save model
model.save('VGG16_model')

#LEARNING CURVE
# Plot training & validation accuracy values
plt.figure(figsize=(10, 5))
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.savefig('/home/zyvasheikh/Desktop/work/PLOTS/VGG16 model accuracy 4.png')


# Plot training & validation loss values
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.savefig('/home/zyvasheikh/Desktop/work/PLOTS/VGG16 model loss 4.png')

#CONFUSION MATRIX
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn import metrics
import seaborn as sns

# Calculate the test_steps_per_epoch
test_steps_per_epoch = np.math.ceil(RGBvalid_batches.samples / RGBvalid_batches.batch_size)

# Predict using the model
predictions = model.predict(RGBvalid_batches, steps=test_steps_per_epoch)

# Get the predicted classes
predicted_classes = np.argmax(predictions, axis=1)
print(predicted_classes)

# Get the true classes and class labels
true_classes = RGBvalid_batches.classes
class_labels = list(RGBvalid_batches.class_indices.keys())

# Print the classification report
report = classification_report(true_classes, predicted_classes, target_names=class_labels)
print(report)

# Print the confusion matrix
conf_matrix = confusion_matrix(true_classes, predicted_classes)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.savefig('/home/zyvasheikh/Desktop/work/PLOTS/VGG16 confusion matrix 4.png')

plt.show(block=True)
