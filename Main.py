import sys
import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image, ImageOps, ImageFilter
import os.path
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

# Step 1: Load the pre-trained model or train the model if it doesn't exist
model_path = 'digit_recognition_model.h5'

if os.path.isfile(model_path):
    model = keras.models.load_model(model_path)
    print('Pre-trained model loaded.')
else:
    print('No pre-trained model found. Training a new model.')
    (x_train, y_train), _ = keras.datasets.mnist.load_data()
    x_train = x_train / 255.0
    x_train = np.expand_dims(x_train, axis=-1)  # Reshape to (num_samples, height, width, channels)

    # Data augmentation
    datagen = keras.preprocessing.image.ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        validation_split=0.2
    )
    train_generator = datagen.flow(x_train, y_train, subset='training')
    validation_generator = datagen.flow(x_train, y_train, subset='validation')

    model = keras.Sequential([
        keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Flatten(),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(train_generator, epochs=10, validation_data=validation_generator)
    model.save(model_path)

# Rest of the code for testing the model remains the same


# Step 2: Test the model with real-time input
if len(sys.argv) < 2:
    print("Please provide the image path as a command-line argument.")
    sys.exit(1)

image_path = sys.argv[1]
image = Image.open(image_path).convert('L')
#to make the background of the given image white
# Threshold value to separate the background from the digits (adjust as needed)
threshold = 100

# Make the background white (pixels below the threshold become white)
image = image.point(lambda x: 255 if x > threshold else x)
# Make the background white (pixels below the threshold become white)

# Invert the image (make the digits black)


image = image.filter(ImageFilter.MedianFilter(size=3))

# Image preprocessing
image = image.resize((28, 28))
image_array = np.array(image)
image_array = image_array / 255.0
image_array = 1 - image_array  # Invert the image (if needed)

image_array = np.expand_dims(image_array, axis=-1)
image_array = np.expand_dims(image_array, axis=0)
# Display the converted image
plt.imshow(image_array[0, :, :, 0], cmap='gray')
plt.title('Converted Image')
plt.axis('off')
plt.show()

# Step 3: Make predictions on the preprocessed image
predictions = model.predict(image_array)
predicted_digit = np.argmax(predictions[0])


print('Predicted Digit:', predicted_digit)

