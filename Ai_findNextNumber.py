import tensorflow as tf
import numpy as np

# Training data
x = np.array([1, 2, 3, 4, 5, 6], dtype=float)
y = np.array([10, 20, 30, 40,50,60], dtype=float)

# Build the model
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(1,)),
    tf.keras.layers.Dense(1)
])

# Compile model
model.compile(optimizer='sgd', loss='mean_squared_error')

# Train
model.fit(x, y, epochs=10, verbose=2)

# User input
z = float(input("Enter the value of z: "))

# FIX: Pass NumPy array instead of list
result = model.predict(np.array([z], dtype=float))

print("Prediction for z →", result)
