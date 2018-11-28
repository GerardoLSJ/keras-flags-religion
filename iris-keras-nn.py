def generate_new_iris_samples():
    new_iris_samples = np.array(
        [[3.2, 1.6, 1.4, 0.5],
         [4.4, 2.2, 4.0, 2.5],
         [5.4, 2.2, 4.2, 1.2],
         [6.0, 3.5, 5.4, 2.2],
         [5.8, 3.1, 5.0, 1.7]], dtype=np.float32)
    return new_iris_samples


def def_new_inputs(new_iris_samples):
    input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": new_iris_samples},
        num_epochs=1,
        shuffle=False)
    return input_fn

"""
    A simple neural network written in Keras (TensorFlow backend) to classify the IRIS data
"""

from keras.models import load_model

import numpy as np
import tensorflow as tf

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

import matplotlib.pyplot as plt

iris_data = load_iris() # load the iris dataset

print('Example data: ')
print(iris_data.data[95:105])
print('Example labels: ')
print(iris_data.target[95:105])

x = iris_data.data
y_ = iris_data.target.reshape(-1, 1) # Convert data to a single column
print('Reshape: ')
print(y_[95:105])

# One Hot encode the class labels
encoder = OneHotEncoder(sparse=False)
y = encoder.fit_transform(y_)
print("encoder:")
print(y[95:105])
a = input()

# Split the data for training and testing
train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.20)

# Build the model

model = Sequential()

model.add(Dense(10, input_shape=(4,), activation='tanh', name='fc1'))
model.add(Dense(10, activation='tanh', name='fc2'))
model.add(Dense(3, activation='sigmoid', name='output'))

# Adam optimizer with learning rate of 0.001
optimizer = Adam(lr=0.001)
model.compile(optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

print('Neural Network Model Summary: ')
print(model.summary())

# Train the model
hist = model.fit(train_x, train_y, verbose=2, batch_size=5, epochs=200)

#hist = model.fit(X_train, Y_train, epochs=1000, verbose=0)
model.save('my_model.h5')  # creates a HDF5 file 'my_model.h5'

plt.figure(figsize=(10,8))
plt.plot(hist.history['acc'], label='Accuracy')
plt.plot(hist.history['loss'], label='Loss')
plt.legend(loc='best')
plt.title('Training Accuracy and Loss')
plt.show()

# Test on unseen data

# z = np.array([[5.1], [3.5], [1.4], [0.2]])
#
# print('Final test set loss: {:4f}'.format(results[0]))
# print('Final test set accuracy: {:4f}'.format(results[1]))

# # new instance where we do not know the answer
# Xnew = iris_data.data[0]
# # make a prediction
# ynew = model.predict_classes([Xnew])
# # show the inputs and predicted outputs
# print("X=%s, Predicted=%s" % (Xnew[0], ynew[0]))
# print('Final test set loss: {:4f}'.format(results[0]))
# print('Final test set accuracy: {:4f}'.format(results[1]))

# # Generate five new iris samples.
# generated_iris_samples = generate_new_iris_samples()

# # Define prediction function inputs.
# predict_input_fn = def_new_inputs(generated_iris_samples)

# # Predict and print the species of newly generated Iris samples
# predictions = list(model.predict(input_fn=predict_input_fn))
# print("Predicted Iris classes: ", [p["classes"][0] for p in predictions])

