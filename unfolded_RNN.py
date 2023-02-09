import numpy as np
import tensorflow as tf

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
model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size)

# Evaluate the model on the test data
test_loss = model.evaluate(x_test, y_test, batch_size=batch_size)

# Make predictions on new data
y_pred = model.predict(x_new)