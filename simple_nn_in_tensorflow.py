import tensorflow as tf
import numpy as np

# Load the Fashion MNIST dataset
fashion_mnist = tf.keras.datasets.fashion_mnist
# split the dataset into training and testing sets
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
# class names for the Fashion MNIST dataset
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']   
# Normalize the pixel values to be between 0 and 1
train_images = train_images / 255.0
test_images = test_images / 255.0

#----------------------------------------------------------------------------------------
# Define the model architecture using Keras Sequential API
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(10)
])
# Compile the model with an optimizer, loss function, and evaluation metric
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), 
              metrics=['accuracy']
)
# Train the model on the training data for 10 epochs
model.fit(train_images, train_labels, epochs=10)
# Evaluate the model on the test data and print the test accuracy
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print("Test accuracy:", test_acc)
probability_model = tf.keras.Sequential([
    model,
    tf.keras.layers.Softmax()
])
# Use the trained model to make predictions on the test images
predictions = probability_model.predict(test_images)
predicted_classes = tf.argmax(predictions, axis=1) # get the index of the maximum probability as the predicted class
print(f"Predicted classes for the first 10 test images: {predicted_classes[:10].numpy()}") # print the predicted classes for the first 10 test images
#------------------------------------------------------------------------------------------

# Define a custom neural network model using the Keras Model subclassing API
class NeuralNetwork(tf.keras.Model):
    # define the layers of the model in the constructor
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.conv1 = tf.keras.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1))
        self.flatten = tf.keras.Flatten()
        self.dense1 = tf.keras.Dense(512, activation='relu')
        self.dense2 = tf.keras.Dense(512, activation='relu')
        self.dense3 = tf.keras.Dense(10, activation='softmax')
    # define the forward pass of the model
    def call(self, x):
        x = self.conv1(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dense2(x)
        logits = self.dense3(x)
        return logits
    
# Create an instance of the custom model and test it with a random input tensor
model = NeuralNetwork()
model.build(input_shape=(None, 28, 28, 1)) # build the model with the specified input shape
X = tf.random.uniform((1, 28, 28, 1)) # random input tensor with shape (1, 28, 28, 1)
logits = model(X) # get the logits from the model   
predictions = tf.nn.softmax(logits) # apply softmax to the logits to get probabilities
y_pred = tf.argmax(predictions, axis=1) # get the index of the maximum probability as the predicted class
print(f"Predicted class: {y_pred.numpy()[0]}") # print the predicted class  
print(f"Predicted probabilities: {predictions.numpy()}") # print the predicted probabilities
print(f"Predicted class name: {class_names[y_pred.numpy()[0]]}") # print the predicted class name
# ------------------------------------------------------------------------------------------------------
