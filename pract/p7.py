import tensorflow as tf 
from tensorflow.keras.applications import MobileNet 
from tensorflow.keras.models import Model 
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D 
from tensorflow.keras.optimizers import Adam 
# Parameters 
IMG_SIZE = 128  # Smaller image size for faster computation 
BATCH_SIZE = 16  # Reduced batch size to save memory 
EPOCHS = 2       # Fewer epochs for quicker training 
LEARNING_RATE = 0.001 
# Load and Preprocess MNIST Dataset 
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data() 
# Use only a subset of the data (e.g., 10,000 samples for training) 
x_train, y_train = x_train[:10000], y_train[:10000] 
x_test, y_test = x_test[:2000], y_test[:2000] 
# Preprocessing function 
def preprocess(image, label): 
    image = tf.image.resize(tf.expand_dims(image, axis=-1), (IMG_SIZE, IMG_SIZE)) / 255.0 
    image = tf.image.grayscale_to_rgb(image)  # Convert grayscale to RGB 
    label = tf.one_hot(label, depth=10)       # One-hot encode labels 
    return image, label 
# Create TensorFlow datasets 
train_dataset = ( 
    tf.data.Dataset.from_tensor_slices((x_train, y_train)) 
    .map(preprocess) 
    .batch(BATCH_SIZE) 
    .prefetch(tf.data.AUTOTUNE) 
) 
test_dataset = ( 
    tf.data.Dataset.from_tensor_slices((x_test, y_test)) 
    .map(preprocess) 
    .batch(BATCH_SIZE) 
    .prefetch(tf.data.AUTOTUNE) 
) 
# Load the smaller pre-trained MobileNet model 
base_model = MobileNet(weights="imagenet", include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3)) 
# Freeze the base model 
base_model.trainable = False 
# Add custom layers on top 
x = base_model.output 
x = GlobalAveragePooling2D()(x)  # Reduce dimensions 
x = Dropout(0.3)(x)              # Dropout for regularization 
predictions = Dense(10, activation="softmax")(x)  # Output layer for 10 classes 
# Create the full model 
model = Model(inputs=base_model.input, outputs=predictions) 
# Compile the model 
model.compile(optimizer=Adam(learning_rate=LEARNING_RATE), 
              loss="categorical_crossentropy", 
              metrics=["accuracy"]) 
# Train the model 
history = model.fit( 
    train_dataset, 
    validation_data=test_dataset, 
    epochs=EPOCHS 
) 
# Evaluate the model on the test dataset 
evaluation = model.evaluate(test_dataset, verbose=1) 
# Print the evaluation metrics 
print(f"Test Loss: {evaluation[0]:.4f}") 
print(f"Test Accuracy: {evaluation[1]:.4f}")