# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf

# Step 1 - Building the CNN

# Initializing the CNN
model = Sequential()

# First convolution layer and pooling
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(120, 120, 1)))
model.add(MaxPooling2D((2, 2)))

# Second convolution layer and pooling
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))

# Third convolution layer
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))

# Flattening the layers
model.add(Flatten())

# Adding a fully connected layer
model.add(Dense(256, activation='relu'))
model.add(Dense(7, activation='softmax'))

# Compiling the CNN
model.compile(optimizer='adam', 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

# Step 2 - Preparing the train/test data and training the model

# Initialize ImageDataGenerators
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=False)

test_datagen = ImageDataGenerator(rescale=1./255)

# Create training set
training_set = train_datagen.flow_from_directory(
    'data/train',
    target_size=(120, 120),
    batch_size=7,
    color_mode='grayscale',
    class_mode='categorical')

# Create test set
test_set = test_datagen.flow_from_directory(
    'data/test',
    target_size=(120, 120),
    batch_size=7,
    color_mode='grayscale',
    class_mode='categorical')

# Debug: Print some information about the data generators
print('Classes in training set:', training_set.class_indices)
print('Classes in test set:', test_set.class_indices)

# Verify data batch generation
for data_batch, labels_batch in training_set:
    print('Data batch shape:', data_batch.shape)  # Should be (batch_size, 120, 120, 1)
    print('Labels batch shape:', labels_batch.shape)  # Should be (batch_size, 7) for one-hot encoded labels
    break  # Only check the first batch

# Train the model
history = model.fit(
    training_set,
    steps_per_epoch=125,  # Adjust according to your dataset
    epochs=7,
    validation_data=test_set,
    validation_steps=50)  # Adjust according to your dataset

# Evaluate the model
test_loss, test_acc = model.evaluate(test_set)
print('Test accuracy: {:2.2f}%'.format(test_acc*100))

# Save entire model to a HDF5 file
model.save('handrecognition_model.hdf5')
model.summary()

# Save model architecture and weights separately
model_json = model.to_json()
with open("gesture-model.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights('gesture-model.h5')
# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf

# Step 1 - Building the CNN

# Initializing the CNN
model = Sequential()

# First convolution layer and pooling
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(120, 120, 1)))
model.add(MaxPooling2D((2, 2)))

# Second convolution layer and pooling
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))

# Third convolution layer
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))

# Flattening the layers
model.add(Flatten())

# Adding a fully connected layer
model.add(Dense(256, activation='relu'))
model.add(Dense(7, activation='softmax'))

# Compiling the CNN
model.compile(optimizer='adam', 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

# Step 2 - Preparing the train/test data and training the model

# Initialize ImageDataGenerators
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=False)

test_datagen = ImageDataGenerator(rescale=1./255)

# Create training set
training_set = train_datagen.flow_from_directory(
    'data/train/data',
    target_size=(120, 120),
    batch_size=7,
    color_mode='grayscale',
    class_mode='categorical')

# Create test set
test_set = test_datagen.flow_from_directory(
    'data/test',
    target_size=(120, 120),
    batch_size=7,
    color_mode='grayscale',
    class_mode='categorical')

# Debug: Print some information about the data generators
print('Classes in training set:', training_set.class_indices)
print('Classes in test set:', test_set.class_indices)

# Verify data batch generation
for data_batch, labels_batch in training_set:
    print('Data batch shape:', data_batch.shape)  # Should be (batch_size, 120, 120, 1)
    print('Labels batch shape:', labels_batch.shape)  # Should be (batch_size, 7) for one-hot encoded labels
    break  # Only check the first batch

# Train the model
history = model.fit(
    training_set,
    steps_per_epoch=125,  # Adjust according to your dataset
    epochs=7,
    validation_data=test_set,
    validation_steps=50)  # Adjust according to your dataset

# Evaluate the model
test_loss, test_acc = model.evaluate(test_set)
print('Test accuracy: {:2.2f}%'.format(test_acc*100))

# Save entire model to a HDF5 file
model.save('handrecognition_model.hdf5')
model.summary()

# Save model architecture and weights separately
model_json = model.to_json()
with open("gesture-model.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights('gesture-model.h5')
