import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_blobs
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ---------------------------------------
# Experiment 18
# Neural Network Analysis for Two Classes
# ---------------------------------------

# Create two-class dataset
X, y = make_blobs(
    n_samples=300,
    centers=[[-2, -2], [2, 2]],
    cluster_std=0.8,
    random_state=42
)

# Split dataset into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.5,
    random_state=42
)

# Create Neural Network
# 2 Hidden Layers
# 3 Neurons in each hidden layer
# Linear activation = identity
# Learning rate = 0.03
model = MLPClassifier(
    hidden_layer_sizes=(3, 3),
    activation='identity',
    solver='sgd',
    learning_rate_init=0.03,
    max_iter=1000,
    random_state=42
)

# Train the model
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

print("Experiment 18")
print("----------------------------")
print("Learning Rate : 0.03")
print("Activation    : Linear")
print("Hidden Layers : 2")
print("Hidden Neurons: 3, 3")
print("Problem Type  : Classification")
print("----------------------------")
print(f"Test Accuracy : {accuracy:.2f}")

# ---------------------------------------
# Create decision boundary
# ---------------------------------------

x_min = X[:, 0].min() - 1
x_max = X[:, 0].max() + 1
y_min = X[:, 1].min() - 1
y_max = X[:, 1].max() + 1

xx, yy = np.meshgrid(
    np.linspace(x_min, x_max, 300),
    np.linspace(y_min, y_max, 300)
)

grid = np.c_[xx.ravel(), yy.ravel()]

# Prediction probabilities
Z = model.predict_proba(grid)[:, 1]

Z = Z.reshape(xx.shape)

# ---------------------------------------
# Plot
# ---------------------------------------

plt.figure(figsize=(10, 7))

# Decision regions
plt.contourf(
    xx,
    yy,
    Z,
    levels=50,
    alpha=0.6,
    cmap='coolwarm'
)

# Training data
plt.scatter(
    X_train[:, 0],
    X_train[:, 1],
    c=y_train,
    cmap='coolwarm',
    edgecolor='black',
    s=40,
    label='Training Data'
)

# Test data
plt.scatter(
    X_test[:, 0],
    X_test[:, 1],
    c=y_test,
    cmap='coolwarm',
    marker='x',
    s=50,
    label='Test Data'
)

plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("Neural Network Analysis for Two Classes")
plt.legend()
plt.grid(True)

plt.show()