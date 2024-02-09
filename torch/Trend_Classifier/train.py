import os
import torch
import torch.nn as nn
import torch.optim as optim
import math
import matplotlib.pyplot as plt
# import data_setup, engine, model_builder, utils

import data_loader

# set hyperparameters
num_epochs = 5
num_batches = 32
hidden_units = 10
learning_rate = 0.001

# Setup directories for loading data

# Target Device
device = "cuda" if torch.cuda.is_available() else "cpu"

# --------------------------------------------------------------------
# Models
# --------------------------------------------------------------------

# Define the binary classification model
class BinaryClassifier(nn.Module):
    def __init__(self):
        super(BinaryClassifier, self).__init__()
        self.hidden = nn.Linear(4, 10)  # 5 input features (o, h, l, c, t)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)
        self.output = nn.Linear(10, 1)

    def forward(self, x):
        x = self.relu(self.hidden(x))
        x = self.output(x)  # Output without sigmoid for numerical stability with BCEWithLogitsLoss
        return x


# --------------------------------------------------------------------
# Evaluation function for accuracy
def calculate_accuracy(y_true, y_pred_logits):
    y_pred = torch.round(torch.sigmoid(y_pred_logits))  # Convert logits to probabilities and then to binary labels
    # print(f"PREDICTIONS {y_true} \n {y_pred}\n")
    correct = (y_pred == y_true).float()  # Compare predicted labels with true labels
    accuracy = correct.sum() / len(correct)  # Calculate accuracy
    return accuracy.item() * 100  # Return accuracy as a percentage


# --------------------------------------------------------------------
# Train the Model
# --------------------------------------------------------------------
model = BinaryClassifier().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-5)

def train(
        model: nn.Module, 
        data: nn.Module,
        optimizer: torch.optim.Optimizer

        ):

    # Training loop

    # Loss function
    loss_fn = nn.BCEWithLogitsLoss()
    # Optimizer (using Adam)
    
    model.train()
    num_epochs = 10001
    # TODO add batch later

    X_train, y_train, X_test, y_test = data
    X_train, y_train = data[0].to(device), data[1].to(device)
    X_test, y_test = data[2].to(device), data[3].to(device)

    loss_array = []
    train_accuracies = []
    test_accuracies = []


    for epoch in range(num_epochs):
        # Forward pass
        outputs = model(X_train).squeeze()  # Squeeze the output to match target shape
        loss = loss_fn(outputs, y_train)
        loss_array.append(loss.item() * 100)
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Print training progress
        if (epoch+1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

        # Evaluation on training data
        model.eval()
        with torch.no_grad():
            train_logits = model(X_train).squeeze()
            train_accuracy = calculate_accuracy(y_train, train_logits)
            train_accuracies.append(train_accuracy)
        
        # Evaluation on test data
        with torch.no_grad():
            test_logits = model(X_test).squeeze()
            test_accuracy = calculate_accuracy(y_test, test_logits)
            test_accuracies.append(test_accuracy)
        
        if (epoch+1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Training Accuracy: {train_accuracy:.2f}%, Test Accuracy: {test_accuracy:.2f}%')
        
    # Plotting
    plt.figure(figsize=(10, 5), facecolor='#212f3d')
    plt.plot(train_accuracies, label='Training Accuracy')
    plt.plot(test_accuracies, label='Test Accuracy')
    plt.plot(loss_array, label='Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Training and Test Accuracy')
    plt.legend()

    ax = plt.gca()
    ax.set_facecolor('#212f3d')
    plt.show()

    return

train(model=model, data=data_loader.load_dataset(), optimizer=optimizer)