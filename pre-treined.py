from keras.models import load_model
from keras.models import load_model

import numpy as np
import tensorflow as tf

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

iris_data = load_iris() # load the iris dataset

#print(iris_data.data[10:25])

model = load_model('my_model.h5')

z = [
[5.8, 4.0, 1.2, 0.2],     #Setosa
[5.8, 2.6, 4.0, 1.2],     #Versicolour
[6.5, 3.0, 5.8, 2.2]      #Virginica 
]
z = np.array(z)
# z = z.transpose()
print('z', z)
# results = model.evaluate(test_x, test_y)
y_prob = model.predict(z) 
y_classes = y_prob.argmax(axis=-1)

values = {
    0: "1 : Setosa",
    1: "2 : Versicolour",
    2: "3 : Virginica",
}
print('model predict: ', list(map(lambda x:values[x] , y_classes)))
