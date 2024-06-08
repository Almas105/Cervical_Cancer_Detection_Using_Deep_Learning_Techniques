import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.preprocessing import LabelEncoder
import numpy as np
import os

# Function to implement Autoencoder (AE) layer
class AE(layers.Layer):
    def __init__(self, latent_dim):
        super(AE, self).__init__()
        self.encoder = tf.keras.Sequential([
            layers.Flatten(),
            layers.Dense(latent_dim, activation='relu'),
        ])
        self.decoder = tf.keras.Sequential([
            layers.Dense(4096, activation='sigmoid'),  # Adjust output shape to match the flattened CNN output
            layers.Reshape((64, 64, 1)),  # Adjust output shape to match the input shape before the first ELM layer
        ])

    def call(self, inputs):
        encoded = self.encoder(inputs)
        decoded = self.decoder(encoded)
        return decoded

# Load and preprocess data with increased data augmentation
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

# Define dataset directory paths
train_dir = r"C:\Users\prasa_o5lltau\OneDrive\Desktop\miniproj\archive\Herlev Dataset\train"
test_dir = r"C:\Users\prasa_o5lltau\OneDrive\Desktop\miniproj\archive\Herlev Dataset\test"

# Load and preprocess train and test data
train_images, train_labels = load_and_preprocess_data(train_dir)
test_images, test_labels = load_and_preprocess_data(test_dir)



# Define shallow CNN model with AE layers
def create_model(input_shape, num_classes):
    model = models.Sequential([
        # Convolutional layer 1
        layers.Conv2D(128, (5, 5), activation='relu', strides=2),
        layers.MaxPooling2D((2, 2), strides=2),
        layers.Flatten(),
        layers.Dense(4096, activation='relu'),
    ])

    # Add AE layer
    model.add(AE(latent_dim=128))
    
    # Flatten the output before feeding to dense layers
    model.add(layers.Flatten())
    
    # Add fully connected layers
    model.add(layers.Dense(4096, activation='relu'))
    model.add(layers.Dense(num_classes, activation='softmax'))
    
    return model

# Create shallow CNN model with AE layers
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
