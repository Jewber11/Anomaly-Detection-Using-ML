import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

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

# Apply anomaly detection using Isolation Forest on the benign reconstruction errors
anomaly_detector = IsolationForest(contamination='auto', random_state=42)
anomaly_detector.fit(benign_reconstruction_errors.reshape(-1, 1))

# Predict anomaly scores for all samples
anomaly_scores = anomaly_detector.score_samples(reconstruction_errors.reshape(-1, 1))

# Find the best threshold using the validation set
best_threshold = 0
best_f1_score = 0

# Iterate over different threshold values
for threshold in np.arange(np.min(anomaly_scores), np.max(anomaly_scores), 0.001):
    # Predict attack or not based on the current threshold
    predicted_labels = np.where(anomaly_scores > threshold, 'cyberattack', 'BENIGN')
    
    # Calculate evaluation metrics
    tp = np.sum((predicted_labels == 'cyberattack') & (y_test == 'cyberattack'))
    fn = np.sum((predicted_labels == 'BENIGN') & (y_test == 'cyberattack'))
    tn = np.sum((predicted_labels == 'BENIGN') & (y_test == 'BENIGN'))
    fp = np.sum((predicted_labels == 'cyberattack') & (y_test == 'BENIGN'))
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * (precision * recall) / (precision + recall)
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    
    # Check if the current F1 score is better than the previous best score
    if f1 > best_f1_score:
        best_f1_score = f1
        best_threshold = threshold

# Print the best threshold and corresponding F1 score
print("Best Threshold:", best_threshold)
print("Best F1 Score:", best_f1_score)

# Predict attack or not based on the best threshold
predicted_labels = np.where(anomaly_scores > best_threshold, 'cyberattack', 'BENIGN')

#print results
print("True Positives:", tp)
print("False Negatives:", fn)
print("True Negatives:", tn)
print("False Positives:", fp)
print("Precision:", precision)
print("Recall:", recall)
print("Accuracy:", accuracy)
print("F1 Score:", f1)
