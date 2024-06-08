import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.preprocessing import LabelEncoder
import numpy as np
import os

# Load and preprocess data
def load_and_preprocess_data(directory, target_size=(224, 224), num_classes=2):
    image_data = []
    labels = []
    label_encoder = LabelEncoder()
    for label in os.listdir(directory):
        label_dir = os.path.join(directory, label)
        for filename in os.listdir(label_dir):
            image_path = os.path.join(label_dir, filename)
            image = tf.keras.preprocessing.image.load_img(image_path, target_size=target_size)
            image = tf.keras.preprocessing.image.img_to_array(image)
            image = image / 255.0
            image_data.append(image)
            labels.append(label)
    image_data = np.array(image_data)
    labels = label_encoder.fit_transform(labels)
    
    # Map labels to the nearest valid label
    labels = np.clip(labels, 0, num_classes - 1)
    
    return image_data, labels

# Load and preprocess train and test data
train_images, train_labels = load_and_preprocess_data(r"C:\Users\prasa_o5lltau\OneDrive\Desktop\miniproj\archive\Herlev Dataset\train")
test_images, test_labels = load_and_preprocess_data(r"C:\Users\prasa_o5lltau\OneDrive\Desktop\miniproj\archive\Herlev Dataset\test")

# Define CaffeNet-like model
model = models.Sequential([
    layers.Conv2D(96, (11, 11), strides=(4, 4), activation='relu', input_shape=(224, 224, 3)),
    layers.MaxPooling2D((3, 3), strides=(2, 2)),
    layers.Conv2D(256, (5, 5), activation='relu', padding='same'),
    layers.MaxPooling2D((3, 3), strides=(2, 2)),
    layers.Conv2D(384, (3, 3), activation='relu', padding='same'),
    layers.Conv2D(384, (3, 3), activation='relu', padding='same'),
    layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((3, 3), strides=(2, 2)),
    layers.Flatten(),
    layers.Dense(4096, activation='relu'),
    layers.Dense(4096, activation='relu'),
    layers.Dense(2, activation='softmax')
])

# Compile the model
model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.01),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(train_images, train_labels, epochs=50, batch_size=20, validation_data=(test_images, test_labels))

# Evaluate the model
test_loss, test_accuracy = model.evaluate(test_images, test_labels)

# Print test accuracy
print("Test Accuracy: {:.2f}%".format(test_accuracy * 100))
