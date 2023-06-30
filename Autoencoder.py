import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split


# Load the CICIDS2017 dataset from a CSV file
# Assuming you have the dataset in a file named 'cicids2017.csv'
dataset = pd.read_csv(r"C:\Users\samgu\OneDrive\Desktop\EE Code\all_data.csv")

# Preprocessing
# Remove any rows with missing values
dataset.dropna(inplace=True)
dataset['Label'] = dataset['Label'].apply(lambda x: 'cyberattack' if x != 'BENIGN' else x)
# Separate features and labels
X = dataset.drop('Label', axis=1)
y = dataset['Label']

# Normalize the feature values using Min-Max scaling
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Impute any missing values using mean imputation
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X_scaled)

# Split the dataset into training, validation, and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=42)
# Define the architecture of the autoencoder
input_dim = X_train.shape[1]  # Number of input features

# Encoder
encoder = keras.models.Sequential([
    keras.layers.Dense(128, activation='relu', input_shape=(input_dim,)),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(32, activation='relu')
])

# Decoder
decoder = keras.models.Sequential([
    keras.layers.Dense(64, activation='relu', input_shape=(32,)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(input_dim, activation='sigmoid')
])

# Combine the encoder and decoder into an autoencoder model
autoencoder = keras.models.Sequential([encoder, decoder])

# Compile the autoencoder model
autoencoder.compile(optimizer='adam', loss='mse')

# Train the autoencoder
autoencoder.fit(X_train, X_train, epochs=10, batch_size=32, validation_data=(X_val, X_val))

# Evaluate the performance of the autoencoder
loss = autoencoder.evaluate(X_test, X_test)
print('Autoencoder loss:', loss)

# Reconstruct input samples
reconstructed_samples = autoencoder.predict(X_test)

# Calculate reconstruction error for each sample
reconstruction_errors = np.mean(np.square(X_test - reconstructed_samples), axis=1)
benign_indices = np.where(y_test == 'BENIGN')[0]
benign_reconstruction_errors = reconstruction_errors[benign_indices]
#average_benign_reconstruction_error = np.mean(benign_reconstruction_errors)
standard_deviation = np.std(benign_reconstruction_errors)
max_benign_reconstruction_error = np.max(benign_reconstruction_errors)
#threshold for attack or not from avg + 1sd of all benign
#threshold = average_benign_reconstruction_error + (standard_deviation/2)
threshold = max_benign_reconstruction_error - standard_deviation
#predict attack or not
predicted_labels = np.where(reconstruction_errors > threshold, 'cyberattack', 'BENIGN')

#metrics
tp = np.sum((predicted_labels == 'cyberattack') & (y_test == 'cyberattack'))
fn = np.sum((predicted_labels == 'BENIGN') & (y_test == 'cyberattack'))
tn = np.sum((predicted_labels == 'BENIGN') & (y_test == 'BENIGN'))
fp = np.sum((predicted_labels == 'cyberattack') & (y_test == 'BENIGN'))

print("True Positives:", tp)
print("False Negatives:", fn)
print("True Negatives:", tn)
print("False Positives:", fp)

# Calculate precision
precision = tp / (tp + fp)

# Calculate recall
recall = tp / (tp + fn)

# Calculate F1-score
f1 = 2 * (precision * recall) / (precision + recall)

# Calculate accuracy
accuracy = (tp + tn) / (tp + tn + fp + fn)

print('f1', {f1})
print('precision', {precision})
print('accuracy', {accuracy})
print('recall', {recall})
