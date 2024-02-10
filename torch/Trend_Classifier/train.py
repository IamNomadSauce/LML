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
def calculate_accuracy(predicted, y_true):
    # Ensure both tensors are on the same device
    y_true = y_true.to(device)
    predicted = predicted.to(device)
    
    # Convert logits to probabilities and then to binary labels
    y_pred = torch.round(torch.sigmoid(predicted))
    
    # Debugging: Print the predicted tensor before and after rounding
    # print(f"Predicted (logits): \n{y_pred}")
    # print(f"Truth (logits): \n{y_true}")
    # print(f"Predicted (rounded): {y_pred} | \n{y_true}")
    
    # Compare predicted labels with true labels
    correct = 0

    print(len(y_pred))

    for v in range(len(y_true)):
        if y_pred[v][0] == y_true[v]:
            correct+=1
        # print(f"Predicted | Truth\n{(y_pred[v][0] == y_true[v]).float()} | {y_pred[v][0]} | {y_true[v]}\n")
    
    # Calculate accuracy
    accuracy = correct / len(y_pred)
    
    # Return accuracy as a percentage
    print(f"ACCURACY | {accuracy}")
    return accuracy * 100


# --------------------------------------------------------------------
# Train the Model
# --------------------------------------------------------------------
model = BinaryClassifier().to(device)
optimizer = optim.Adam(model.parameters(), lr=5e-3, weight_decay=5e-5)

def evaluate(train, test):

    # Evaluation on training data
    model.eval()
    with torch.no_grad():
        train_logits = model(train).squeeze()
        train_accuracy = calculate_accuracy(test, train_logits)
    return train_accuracy

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
        
        train_accuracy = evaluate(X_train, y_train)
        test_accuracy = evaluate(X_test, y_test)

        train_accuracies.append(train_accuracy)
        test_accuracies.append(test_accuracy)
        
        if (epoch+1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Training Accuracy: {train_accuracy:.2f}%, Test Accuracy: {test_accuracy:.2f}%')
    
    # Save Model and Weights
    torch.save(model.state_dict(), './weights/model_weights.pth')        
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

def load_weights(model: nn.Module, input_data, validation_data):
    model.load_state_dict(torch.load('./weights/model_weights.pth', map_location=device))
    model.eval()
    predictions = []
    with torch.no_grad():
        input_data = input_data.to(device)
        predictions = model(input_data)
        inf_accuracy = calculate_accuracy(predictions, validation_data)
        print(f"Inference Accuracy {inf_accuracy}%")
    
    # print(f"Predictions: {predictions}\nTruth: {validation_data}")

# train(model=model, data=data_loader.load_dataset("BTCUSD"), optimizer=optimizer)

X_train, y_train, X_test, y_test = data_loader.load_dataset("DOGEUSD")
load_weights(model, X_train, y_train)