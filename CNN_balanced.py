import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import *
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.utils import to_categorical

# Load the dataset
data = pd.read_csv(r"C:\Users\samgu\OneDrive\Desktop\EE code\all_data.csv")

# Extract features and labels
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# Encode the categorical labels
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

# Split the selected dataset into train, validation, and test sets (70%, 15%, 15%)
X_train, X_test, y_train, y_test = train_test_split(X_selected, y_selected, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=42)

# Convert input data to float32
X_train = X_train.astype('float32')
X_val = X_val.astype('float32')
X_test = X_test.astype('float32')

# Reshape the data for 2D CNN
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1, 1))
X_val = X_val.reshape((X_val.shape[0], X_val.shape[1], 1, 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1, 1))

# Convert labels to one-hot encoding
num_classes = len(label_encoder.classes_)
y_train = to_categorical(y_train, num_classes)
y_val = to_categorical(y_val, num_classes)
y_test = to_categorical(y_test, num_classes)

# Build the CNN model
model = Sequential()
model.add(Conv2D(32, (3, 1), activation='relu', input_shape=(X_train.shape[1], 1, 1)))
model.add(Conv2D(32, (3, 1), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 1)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=1, batch_size=32)

# Evaluate the model on test data
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_test_classes = np.argmax(y_test, axis=1)

# Calculate evaluation metrics
accuracy = accuracy_score(y_test_classes, y_pred_classes)
precision = precision_score(y_test_classes, y_pred_classes, average='weighted')
recall = recall_score(y_test_classes, y_pred_classes, average='weighted')
f1 = f1_score(y_test_classes, y_pred_classes, average='weighted')
cr = classification_report(y_test_classes, y_pred_classes)
cm = confusion_matrix(y_test_classes, y_pred_classes)
tn = cm[0, 0]
fp = cm[0, 1:].sum()
fn = cm[1:, 0].sum()
tp = cm[1:, 1:].sum()

print(f'Test Accuracy: {accuracy:.4f}')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1 Score: {f1:.4f}')
print(f'Classification Report:\n{cr}')
print(f'True Negatives: {tn}')
print(f'False Positives: {fp}')
print(f'False Negatives: {fn}')
print(f'True Positives: {tp}')
