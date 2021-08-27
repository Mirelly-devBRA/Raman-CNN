# This model is the same as the other one, just using the sequential API
# The keras documentation explains why you may want to to this vs the 
# functional API i used

model = Sequential([
    # InputLayer(input_shape=(913,2)),
    Conv1D(filters=16, kernel_size=21, input_shape=(913,2)),
    BatchNormalization(),
    LeakyReLU(),
    MaxPool1D(pool_size=2),

    Conv1D(filters=32, kernel_size=11),
    BatchNormalization(),
    LeakyReLU(),
    MaxPool1D(pool_size=2),

    Conv1D(filters=64, kernel_size=5 ),
    BatchNormalization(),
    LeakyReLU(),
    MaxPool1D(pool_size=2),

    Flatten(),

    Dense(units=2048, activation='tanh'),
    BatchNormalization(),
    Dropout(rate = 0.5),

    Dense(units=num_classes),
    BatchNormalization(),
    Softmax(),
])

