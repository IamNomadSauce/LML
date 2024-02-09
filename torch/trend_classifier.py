import torch
import torch.nn as nn
import torch.optim as optim
import mysql.connector
import math
import matplotlib.pyplot as plt
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

# Assuming your dataset is already loaded and preprocessed
# X_train, y_train, X_test, y_test are your training and testing data tensors

device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = 8

# Build trendlines by hand for validation
def make_trendlines(candles):
    trendlines = []
    ts = 1
    trend_state = []

    for index, candle in enumerate(candles):
        # print("CANDLE", index, candle) # [t, o, h, l, c, v]
        if index == 0:
            current = {
                "Time": candles[0][0],
                "Point": candles[0][2],
                "Inv": candles[0][3],
                "Direction": -1 if candles[0][2] < candles[0][3] else 1,
            }
            trendlines.append(current)
            trend_state.append(-1 if candles[0][2] < candles[0][3] else 1)
            pass
        else:
            # Higher High in uptrend  (continuation)
            if candle[2] > current["Point"] and current["Direction"] == 1:
                current = {
                    "Time": candle[0],
                    "Point": candle[2],
                    "Inv": candle[3],
                    "Direction": 1,
                }
                ts = 1

                
            # Higher High in downtrend  (new trend)
            if (candle[2] > current["Inv"]  and current["Direction"] == 0):
                current = {
                    "Time": candle[0],
                    "Point": candle[2],
                    "Inv": candle[3],
                    "Direction": 0 if candle[2] < candle[3] else 1,
                }
                ts = 1
                
            # Lower Low in uptrend (new trend)
            if (candle[3] < current["Inv"]  and current["Direction"] == 1):
                current = {
                    "Time": candle[0],
                    "Point": candle[3],
                    "Inv": candle[2],
                    "Direction": 0 if candle[2] < candle[3] else 1,
                }
                ts = 0

                
                
            # Lower Low in downtrend  (continuation)
            if candle[3] < current["Point"] and current["Direction"] == 0:
                current = {
                    "Time": candle[0],
                    "Point": candle[3],
                    "Inv": candle[2],
                    "Direction": 0 if candle[2] < candle[3] else 1,
                }
                ts = 0


            # Janky fix to get over skipping trends that have the same unique time for start and end
            # if current["StartTime"] == current["EndTime"]:
            #     current["StartTime"] -= 1
            trendlines.append(current)
            # print("CANDLE LENGTH", len(current))
            # print("TREND", ts)
            trend_state.append(ts)
            
        # print("Trendline", current, "\n")


    return trend_state

def mean_normalize(data):
    mean_vals = torch.mean(data, dim=0)
    range_vals = torch.max(data, dim=0).values - torch.min(data, dim=0).values
    return (data - mean_vals) / range_vals
    
def load_dataset():
    connection = mysql.connector.connect(
        host="localhost",
        user="root",
        password="1234567",
        database="markets"
    )

    cursor = connection.cursor()

    candles_query = "SELECT * FROM coinbase_BTCUSD_1 ORDER BY time"
    # trends_query = "SELECT * FROM coinbase_BTCUSD_1_trendlines"

    cursor.execute(candles_query)
    candles = cursor.fetchall()
    # Limit to 1000 candles
    candles = candles[:1000]
    # cursor.execute(trends_query)
    # trends = cursor.fetchall()
    trends = make_trendlines(candles)

    # print(f"TRENDS {trends}")


    cursor.close()
    connection.close()
    # print(f"Candles length {len(candles)} \n", f"TRENDS LENGTH {len(trends)} \n" )
    
    # Correctly split the data into training and test sets
    split_index = math.floor(len(candles) * 0.9)
    candles_training = candles[:split_index]
    candles_test = candles[split_index:]

    trends_training = trends[:split_index]
    trends_test = trends[split_index:]

    
    # print(f"SPLIT DATA \n {len(trends_training)} \n {len(trends_test)} \n")

    candles_training = [[candle[1], candle[2], candle[3], candle[4]] for candle in candles_training]
    candles_test = [[candle[1], candle[2], candle[3], candle[4]] for candle in candles_test]

    # Convert candles and trends data to tensors
    X_train_tensor = torch.tensor(candles_training, dtype=torch.float)
    y_train_tensor = torch.tensor(trends_training, dtype=torch.float)
    X_test_tensor = torch.tensor(candles_test, dtype=torch.float)
    y_test_tensor = torch.tensor(trends_test, dtype=torch.float)

    return X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor

X_train, y_train, X_test, y_test = load_dataset()
# Apply mean normalization to the input data
X_train = mean_normalize(X_train)
X_test = mean_normalize(X_test)
X_train, y_train = X_train.to(device), y_train.to(device)
X_test, y_test = X_test.to(device), y_test.to(device)


# Define the binary classification model
class BinaryClassifier(nn.Module):
    def __init__(self):
        super(BinaryClassifier, self).__init__()
        self.hidden = nn.Linear(4, 10)  # 5 input features (o, h, l, c, t)
        self.relu = nn.ReLU()
        self.output = nn.Linear(10, 1)

    def forward(self, x):
        x = self.relu(self.hidden(x))
        # print(f"Forward 1 {x}")
        x = self.output(x)  # Output without sigmoid for numerical stability with BCEWithLogitsLoss
        # print(f"Forward 2 {x} \n")

        return x

# Evaluation function for accuracy
def calculate_accuracy(y_true, y_pred_logits):
    y_pred = torch.round(torch.sigmoid(y_pred_logits))  # Convert logits to probabilities and then to binary labels
    correct = (y_pred == y_true).float()  # Compare predicted labels with true labels
    # print("\nPredicted", y_pred, "\nCorrect", correct, len(y_pred), len(correct))
    accuracy = correct.sum() / len(correct)  # Calculate accuracy
    return accuracy.item() * 100  # Return accuracy as a percentage

# Initialize the model
model = BinaryClassifier().to(device)

# Loss function
loss_fn = nn.BCEWithLogitsLoss()

# Optimizer (using Adam)
optimizer = optim.Adam(model.parameters(), lr=0.1, weight_decay=0.05)

train_accuracies = []
test_accuracies = []
# Create TensorDataset objects for both training and test sets
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

# Create DataLoaders for both training and test sets
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
# Training loop
num_epochs = 1001
for epoch in range(num_epochs):
    model.train()  # Set the model to training mode
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Forward pass
        outputs = model(inputs).squeeze()
        loss = loss_fn(outputs, targets)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch+1) % 100 == 0 and batch_idx == len(train_loader) - 1:
            # Print training progress
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

    # Evaluation on training and test data
    if (epoch+1) % 100 == 0:
        model.eval()  # Set the model to evaluation mode
        with torch.no_grad():
            # Calculate accuracy on the training set
            train_correct = 0
            train_total = 0
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs).squeeze()
                predicted = torch.round(torch.sigmoid(outputs))
                train_total += targets.size(0)
                train_correct += (predicted == targets).sum().item()
            train_accuracy = 100 * train_correct / train_total

            # Calculate accuracy on the test set
            test_correct = 0
            test_total = 0
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs).squeeze()
                predicted = torch.round(torch.sigmoid(outputs))
                test_total += targets.size(0)
                test_correct += (predicted == targets).sum().item()
            test_accuracy = 100 * test_correct / test_total

            print(f'Epoch [{epoch+1}/{num_epochs}], Training Accuracy: {train_accuracy:.2f}%, Test Accuracy: {test_accuracy:.2f}%')

# Final evaluation on training data
model.eval()  # Set the model to evaluation mode
with torch.no_grad():
    train_logits = model(X_train).squeeze()
    train_accuracy = calculate_accuracy(y_train, train_logits)

    print(f'Training Accuracy: {train_accuracy:.2f}% | {train_logits}')

# Final evaluation on test data
with torch.no_grad():
    test_logits = model(X_test).squeeze()
    test_accuracy = calculate_accuracy(y_test, test_logits)
    print(f'Test Accuracy: {test_accuracy:.2f}% | {test_logits}')



# Plotting
plt.figure(figsize=(10, 5), facecolor='#212f3d')
plt.plot(train_accuracies, label='Training Accuracy')
plt.plot(test_accuracies, label='Test Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title('Training and Test Accuracy')
plt.legend()

ax = plt.gca()
ax.set_facecolor('#212f3d')

plt.show()