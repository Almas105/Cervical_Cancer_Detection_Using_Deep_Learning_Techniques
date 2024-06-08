import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.preprocessing import LabelEncoder
import numpy as np
import os

# Function to implement ELM layer
class ELM(layers.Layer):
    def __init__(self, num_hidden_units, input_shape):
        super(ELM, self).__init__()
        self.num_hidden_units = num_hidden_units
        self.random_weights = tf.Variable(tf.random.normal(shape=(input_shape, num_hidden_units)), trainable=False)

    def call(self, inputs):
        # Compute hidden layer output
        hidden_output = tf.matmul(inputs, self.random_weights)
        hidden_output = tf.nn.relu(hidden_output)
        return hidden_output

# Load and preprocess data with increased data augmentation
def load_and_preprocess_data(directory, target_size=(227, 227), num_classes=2):
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

# Define dataset directory paths
train_dir = r"C:\Users\prasa_o5lltau\OneDrive\Desktop\miniproj\archive\Herlev Dataset\train"
test_dir = r"C:\Users\prasa_o5lltau\OneDrive\Desktop\miniproj\archive\Herlev Dataset\test"

# Load and preprocess train and test data
train_images, train_labels = load_and_preprocess_data(train_dir)
test_images, test_labels = load_and_preprocess_data(test_dir)

# Define CaffeNet-like model with ELM layers
def create_model(input_shape, num_classes):
    model = models.Sequential([
        layers.Conv2D(96, (11, 11), strides=(4, 4), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((3, 3), strides=(2, 2)),
        layers.Conv2D(256, (5, 5), padding='same', activation='relu'),
        layers.MaxPooling2D((3, 3), strides=(2, 2)),
        layers.Conv2D(384, (3, 3), padding='same', activation='relu'),
        layers.Conv2D(384, (3, 3), padding='same', activation='relu'),
        layers.Conv2D(256, (3, 3), padding='same', activation='relu'),
        layers.MaxPooling2D((3, 3), strides=(2, 2)),
        layers.Flatten(),
        layers.Dense(4096, activation='relu'),
        layers.Dense(4096, activation='relu'),
    ])

    # Add first ELM layer
    model.add(ELM(num_hidden_units=2048, input_shape=4096))
    
    # Add second ELM layer
    model.add(ELM(num_hidden_units=num_classes, input_shape=2048))
    
    return model

# Create CaffeNet-like model with ELM layers
model = create_model(input_shape=train_images.shape[1:], num_classes=2)

# Adjust learning rate and optimizer
opt = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9)

# Compile the model with appropriate loss function
model.compile(optimizer=opt,
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model with specified number of epochs (50)
history = model.fit(train_images, train_labels, epochs=50, batch_size=20, validation_data=(test_images, test_labels))

# Evaluate the model
test_loss, test_accuracy = model.evaluate(test_images, test_labels)

# Print test accuracy
print("Test Accuracy: {:.2f}%".format(test_accuracy * 100))
