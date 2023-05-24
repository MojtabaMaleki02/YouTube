import tensorflow as tf

# Load and preprocess the MNIST dataset
mnist = tf.keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Normalize pixel values to be between 0 and 1
train_images = train_images / 255.0
test_images = test_images / 255.0

# Define the model architecture
model = tf.keras.Sequential([
    # Flattens input images into a 1-dimensional array
    tf.keras.layers.Flatten(input_shape=(28, 28)),  
    # Fully connected layer with 128 units and ReLU activation
    tf.keras.layers.Dense(128, activation='relu'),  
    # ReLU is a activation function and it's stands for Rectified Linear Unit
    # Fully connected layer with 10 units (output classes) and softmax activation
    tf.keras.layers.Dense(10, activation='softmax')  
])

# Compile the model
# Adam optimizer for training the model
model.compile(optimizer='adam',  
              # Loss function for training the model
              loss='sparse_categorical_crossentropy',  
              # Metric used to evaluate the model during training
              metrics=['accuracy'])  

# Train the model
# Train the model on the training data for 5 epochs
# epoch is the training of the neural network with all the training data for one cycle
model.fit(train_images, train_labels, epochs=5)  

# Evaluate the model
# Evaluate the model on the test data
test_loss, test_accuracy = model.evaluate(test_images, test_labels) 
# Print the test accuracy
print(f'Test accuracy: {test_accuracy}')  

# Make predictions
# Generate predictions for the test images
predictions = model.predict(test_images)  
