import numpy as np
from hmmlearn import hmm
# Step 1: Prepare the data
# Observations: 0 = No Umbrella, 1 = Umbrella
observations = np.array([[0], [0], [1], [1], [0], [1], [0], [1]])
# States: 0 = Sunny, 1 = Rainy
# The hidden state sequence is not provided, we will train the HMM model to predict it
# We will only use the observation sequence for training.

# Step 2: Define and Train the HMM Model
# Define the HMM: 2 hidden states (Sunny, Rainy), 2 possible observations (No Umbrella, Umbrella)
model = hmm.MultinomialHMM(n_components=2, n_iter=1000)
# Train the model on the observations (without the hidden state labels)
model.fit(observations)

# Step 3: Predict the hidden states (weather) given the observations
predicted_states = model.predict(observations)

# Step 4: Output results
print("Predicted Hidden States (Weather):")
print(predicted_states)

# Step 5: Output the model's parameters (transition matrix, emission matrix)
print("\nTransition Matrix (State to State):")
print(model.transmat_)
print("\nEmission Matrix (Observation | State):")
print(model.emissionprob_)

# Step 6: Making predictions for new observations (example)
# Let's predict the weather based on a new sequence of umbrella usage
new_observations = np.array([[0], [1], [1]])  # No Umbrella, Umbrella, Umbrella
predicted_new_states = model.predict(new_observations)
print("\nPredicted Hidden States for New Observations (No Umbrella, Umbrella, Umbrella):")
print(predicted_new_states)

