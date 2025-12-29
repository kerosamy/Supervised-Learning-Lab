import os
import struct
import warnings
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from tensorflow.keras import layers, models, regularizers

# --- 0. SUPPRESS WARNINGS ---
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
warnings.filterwarnings("ignore")

# --- 1. DATA LOADING ---
def read_idx(filename):
    with open(filename, 'rb') as f:
        zero, data_type, dims = struct.unpack('>HBB', f.read(4))
        shape = tuple(struct.unpack('>I', f.read(4))[0] for d in range(dims))
        return np.frombuffer(f.read(), dtype=np.uint8).reshape(shape)

data_path = 'data'
try:
    X_train_raw = read_idx(os.path.join(data_path, 'train-images.idx3-ubyte'))
    y_train_raw = read_idx(os.path.join(data_path, 'train-labels.idx1-ubyte'))
    X_test_raw = read_idx(os.path.join(data_path, 't10k-images.idx3-ubyte'))
    y_test_raw = read_idx(os.path.join(data_path, 't10k-labels.idx1-ubyte'))
except Exception as e:
    print(f"Error loading files: {e}")
    exit()

# Combine for preprocessing
X_all = np.concatenate((X_train_raw, X_test_raw), axis=0).reshape(-1, 784)
y_all = np.concatenate((y_train_raw, y_test_raw), axis=0)

# --- 2. PREPROCESSING & CLEANING ---
# Normalization
X_norm = X_all / 255.0

# Duplicate Removal
df = pd.DataFrame(X_norm)
df['label'] = y_all
df = df.drop_duplicates()
X_clean = df.drop('label', axis=1).values
y_clean = df['label'].values
print(f"Cleaned Data: {len(X_clean)} unique samples.")

# Feature Selection (Variance Threshold)
selector = VarianceThreshold(threshold=(.99 * (1 - .99)))
X_selected = selector.fit_transform(X_clean)
print(f"Features reduced from 784 to {X_selected.shape[1]}")

# 70/15/15 Split (Random Seed 42)
X_train, X_temp, y_train, y_temp = train_test_split(
    X_selected, y_clean, train_size=0.70, random_state=42, stratify=y_clean
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
)

# Helper function for reporting
def report(name, y_true, y_pred):
    print(f"\n{'='*20} {name} {'='*20}")
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    print(classification_report(y_true, y_pred, digits=4))
    print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))

# --- 3. TUNING & SOFTMAX REGRESSION ---
print("\n--- Tuning Softmax (using Validation Set) ---")
c_values = [0.1, 1.0, 10.0]
best_c = 1.0
best_val_acc = 0

for c in c_values:
    model = LogisticRegression(solver='saga', penalty='l2', C=c, max_iter=20, random_state=42)
    model.fit(X_train, y_train)
    v_acc = accuracy_score(y_val, model.predict(X_val))
    if v_acc > best_val_acc:
        best_val_acc = v_acc
        best_c = c

# Final Softmax Models
print(f"Best C selected: {best_c}")
for p in [None, 'l1', 'l2']:
    name = f"Softmax (Penalty: {p})"
    sm = LogisticRegression(solver='saga', penalty=p, C=best_c, max_iter=50, random_state=42)
    sm.fit(X_train, y_train)
    report(name, y_test, sm.predict(X_test))

# --- 4. TUNING & NEURAL NETWORK ---
print("\n--- Tuning Neural Network (using Validation Set) ---")
lr_values = [0.001, 0.0001]
best_lr = 0.001
best_nn_val = 0

for lr in lr_values:
    m = models.Sequential([
        layers.Input(shape=(X_selected.shape[1],)),
        layers.Dense(128, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])
    m.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), 
              loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    m.fit(X_train, y_train, epochs=3, verbose=0)
    _, acc = m.evaluate(X_val, y_val, verbose=0)
    if acc > best_nn_val:
        best_nn_val = acc
        best_lr = lr

print(f"Best Learning Rate selected: {best_lr}")

def build_final_nn(reg=None):
    model = models.Sequential([
        layers.Input(shape=(X_selected.shape[1],)),
        layers.Dense(128, activation='relu', kernel_regularizer=reg),
        layers.Dense(64, activation='relu', kernel_regularizer=reg),
        layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=best_lr), 
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Final NN Models
for reg_type, r_val in [("Only", None), ("Lasso", regularizers.l1(0.0001)), ("Ridge", regularizers.l2(0.0001))]:
    nn = build_final_nn(reg=r_val)
    nn.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=64, verbose=0)
    preds = np.argmax(nn.predict(X_test), axis=1)
    report(f"Neural Network ({reg_type})", y_test, preds)