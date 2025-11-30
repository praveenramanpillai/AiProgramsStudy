import tensorflow as tf
import numpy as np

# Set a random seed for reproducible results
tf.random.set_seed(42)
np.random.seed(42)

# Training data: y = 10 * x
x = np.array([1, 2, 3, 4, 5, 6], dtype=float)
y = np.array([10, 20, 30, 40, 50, 60], dtype=float)

# 1. Build the model: Simple Linear Regression Model (y = Wx + b)
model = tf.keras.Sequential([
    # Input layer
    tf.keras.layers.Input(shape=(1,)),
    # Dense Layer (Hidden/Output)
    tf.keras.layers.Dense(1)
])


# 2. Compile model
model.compile(optimizer='sgd', loss='mean_squared_error') # Using Stochastic Gradient Descent (sgd) and Mean Squared Error (mse) loss.

# 3. Train
print("--- Starting Training ---")
model.fit(x, y, epochs=1000, verbose=1)  # 1000 iterations of training
print("--- Training Finished ---")

# 4. Prediction
try:
    z = float(input("\nEnter the value of z for prediction (e.g., 7.0 or 10.5): ")) # user input for prediction
    z_input = np.array([z], dtype=float)

    result = model.predict(z_input, verbose=0) # Predict the value

    print(f"\nPrediction for z ({z}) → {result[0][0]:.4f}") # # Print the prediction, accessing the single value in the 2D array result

except ValueError:
    print("Invalid input. Please enter a numerical value.")