
"""
    A simple neural network written in Keras (TensorFlow backend) to classify the IRIS data
"""

import numpy as np

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras import optimizers
import matplotlib.pyplot as plt
# iris_data = load_iris() # load the iris dataset

flag_data = []
y = []
with open("data_flags.csv") as f:
    i = 0
    for line in f:
        a = line.split(',')
        flag_data.append(a[1:5] + a[7:16] + a[18:28])
        # print(len(a[1:5] + a[7:9] + a[18:28]))
        # print(a[1:5] + a[7:9] + a[18:28])
        # input()
        y.append(a[6])


flag_data = np.array(flag_data)
y = np.array(y)

print('flag_data[0]', flag_data[0])
print('y[0]', y)

x = flag_data
y_ = y.reshape(-1, 1) # Convert data to a single column

# print("x, y", x, y_)

# One Hot encode the class labels
encoder = OneHotEncoder(sparse=False)
y = encoder.fit_transform(y_)
print("encoder")
print(y)
# Split the data for training and testing
train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.20)

# Build the model

model = Sequential()

model.add(Dense(23, input_shape=(23,), activation='relu', name='fc1'))
model.add(Dense(100, activation='tanh', name='fc2'))
model.add(Dense(8, activation='sigmoid', name='output'))

# Adam optimizer with learning rate of 0.001
#optimizer = Adam(lr=0.001)
#model.compile(optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# sgd = optimizers.SGD(lr=0.01, clipnorm=1.)
# model.compile(loss='mean_squared_error', optimizer='sgd', metrics=['accuracy'])

#with mean_squared_error
rms = optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)
#60%
nadam = optimizers.Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)
#60%
adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
#69
adamax = optimizers.Adamax(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0)
#61
adagrad = optimizers.Adagrad(lr=0.01, epsilon=None, decay=0.0)
# model.compile(loss='mean_squared_error', optimizer=adamax, metrics=['accuracy'])

#with categorical_crossentropy
#adamax - 69

# poisson
#adamax - 63
model.compile(loss='categorical_crossentropy', optimizer=adamax, metrics=['accuracy'])

print('Neural Network Model Summary: ')
print(model.summary())

# Train the model  batch_size=5,
hist = model.fit(train_x, train_y, verbose=2,  batch_size=5, epochs=2000)
model.save('trainedFlags.h5')  # creates a HDF5 file 'my_model.h5'

plt.figure(figsize=(10,8))
plt.plot(hist.history['acc'], label='Accuracy')
plt.plot(hist.history['loss'], label='Loss')
plt.legend(loc='best')
plt.title('Training Accuracy and Loss')
plt.show()

# Test on unseen data

# results = model.predict(test_x)
# print(results)