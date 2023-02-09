import numpy as np
import tensorflow as tf

# prepare data
data = "" # the path of the data for evaluation
training_data_size = int(data.shape[0] * 0.7)
validation_data_size = int(data.shape[0] * 0.15)
test_data_size = data.shape[0] - training_data_size - validation_data_size
training_data = data[:training_data_size, :]
validation_data = data[training_data_size:training_data_size+validation_data_size, :]
test_data = data[training_data_size+validation_data_size:, :]


# Define the number of hidden units
hidden_unit = 32

# Define the number of epochs
epochs = 100

# Define the batch size
batch_size = 1

# Define the input shape
input_shape = (timesteps, input_dim)

# Define the model
model = tf.keras.models.Sequential([
    tf.keras.layers.SimpleRNN(hidden_unit, activation='relu', input_shape=input_shape, return_sequences=True),
    tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(1))
])

# Compile the model
model.compile(optimizer='sgd', loss='mean_squared_error')

# Fit the model to the training data
model.fit(training_data, epochs=epochs, batch_size=batch_size, validation_data=(validation_data,))

# Evaluate the model on the test data
test_loss = model.evaluate(test_data)