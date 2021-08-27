import numpy as np
import pandas as pd
import glob
import os
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split 

import tensorflow as tf

from keras.utils import to_categorical
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Conv1D, Flatten, LeakyReLU, MaxPool1D, Concatenate, Dropout, BatchNormalization, Softmax, InputLayer
from keras.utils.vis_utils import plot_model

#---------------------- START HERE ----------------------#

x_data = []
y_data=[]
# File path to where data is stored
data_loc = 'C:/Users/david/Documents/ISI Placement/ISI/ISI_Dataset'
files = glob.glob("C:/Users/david/Documents/ISI Placement/ISI/ISI_Dataset/*/*.pspec")

# List of Arrays
for file in files:
    data = np.loadtxt(file, comments='#', delimiter='\t', skiprows=100, 
                        dtype=str, usecols=[1,2])
    x_data.append(data)
    
for file in files:
    y_data = list(map(lambda x: os.path.split(os.path.split(x)[0])[1],files))

x_data = np.stack(x_data)
x_data = x_data.astype(float)


#---------------------- ONE HOT ENCODING ----------------------#

le = LabelEncoder()
le.fit(y_data)
classes = le.classes_
num_classes = len(classes)
y_data = le.transform(y_data)
y_data = to_categorical(y_data, num_classes)


#---------------------- DATA PREPROCESSING ----------------------#

x_train, x_test, y_train, y_test =  train_test_split(x_data,
                                                    y_data,
                                                    test_size=.2)

def CNN_model():
    inputs = Input(shape=(913,2))
    x =  Conv1D(filters=16, kernel_size=21)(inputs)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = MaxPool1D(pool_size=2)(x)
    
    x = Conv1D(filters=32, kernel_size=11)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = MaxPool1D(pool_size=2)(x)
    
    x = Conv1D(filters=64, kernel_size=5 )(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = MaxPool1D(pool_size=2)(x)
    
    x = Flatten()(x)
    
    x = Dense(units=2048, activation='tanh')(x)
    x = BatchNormalization()(x)
    x = Dropout(rate = 0.5)(x)
    
    x = Dense(units=num_classes)(x)
    x = BatchNormalization()(x)
    outputs = Softmax()(x)
    
    model = Model(inputs=inputs, outputs=outputs, name="Raman_CNN")
    
    model.compile(optimizer='adam', loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    return model


model = CNN_model()

#If you wish to load a specific sets of weights, do it here -  
# path_to_weights ='C:/Users/david/Documents/ISI Placement/ISI/Weights/my_model.h5'
# model.load_weights(path_to_weights)

model.summary()
plot_model(model, to_file='model_plot.png', show_shapes=True)
history = model.fit(x_train,
                    y_train,
                    batch_size=32,
                    epochs=50,
                    validation_split=0.1)


test_scores = model.evaluate(x_test, y_test, verbose=2)
print("Test loss:", test_scores[0])
print("Test accuracy:", test_scores[1])


model.save_weights("C:/Users/david/Documents/ISI Placement/ISI/Weights/my_model.h5", save_format='h5')



loss, accuracy = model.evaluate(x_test, y_test, verbose=False)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.legend(['training', 'validation'], loc='best')
plt.show()

predictions = model.predict(x_test)

font_dict = {'color': 'black'}
for i in range(len(x_test)):
    plt.plot(x_test[i, :, 0], x_test[i, :, 1])
    prediction = le.inverse_transform([np.argmax(predictions[i])])
    true_value = le.inverse_transform([np.argmax(y_test[i])])
    font_dict['color'] = "black" if prediction == true_value else 'red'
    plt.title(
        f"Predicted: {prediction}, Truth: {true_value}", fontdict=font_dict)
    plt.tight_layout()
    plt.show()

