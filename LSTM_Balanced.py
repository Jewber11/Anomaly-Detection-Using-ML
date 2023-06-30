import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import *
import matplotlib.pyplot as plt
import numpy as np

# Load the dataset
df = pd.read_csv(r"C:\Users\samgu\OneDrive\Desktop\all_data.csv")
# Extract the features and labels
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# Perform label encoding if needed
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Split the dataset into benign and attack samples
benign_indices = np.where(y == 0)[0]
attack_indices = np.where(y == 1)[0]

# Randomly select 30% of the benign samples
num_benign_samples = int(0.3 * len(benign_indices))
random_benign_indices = np.random.choice(benign_indices, size=num_benign_samples, replace=False)

# Concatenate the selected benign samples with the attack samples
selected_indices = np.concatenate((random_benign_indices, attack_indices))
X_selected = X[selected_indices]
y_selected = y[selected_indices]

# Split the dataset into training, validation, and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=42)

# Create the LSTM model
model = Sequential()
model.add(LSTM(128, input_shape=(X_train.shape[1], 1), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(64))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Reshape the input data for LSTM
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_val = X_val.reshape((X_val.shape[0], X_val.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# Train the model
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_val, y_val))

# Make predictions on the test set
y_pred = model.predict(X_test)
y_pred = np.round(y_pred).flatten()

# Compute the multilabel confusion matrix
cm = confusion_matrix(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted')
print(f'F1 Score: {f1:.4f}')
print('Classification Report:\n', classification_report(y_test, y_pred))
tn = cm[0, 0]  
fp = cm[0, 1:].sum()  
fn = cm[1:, 0].sum()  
tp = cm[1:, 1:].sum()  

print(f'True Negatives: {tn}')
print(f'False Positives: {fp}')
print(f'False Negatives: {fn}')
print(f'True Positives: {tp}')