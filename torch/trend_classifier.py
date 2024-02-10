import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim as optim
import mysql.connector
import math
import matplotlib.pyplot as plt
import numpy as np

# Assuming your dataset is already loaded and preprocessed
# X_train, y_train, X_test, y_test are your training and testing data tensors

device = "cuda" if torch.cuda.is_available() else "cpu"
print("DEVICE: ", device)

# Build trendlines by hand for validation
def make_trendlines(candles):
    trendlines = []
    ts = 1
    trend_state = []

    for index, candle in enumerate(candles):
        # print("CANDLE", candle)
        if candle[4] > candle[1]:
            # print("HIGHER")
            ts = 1
        else:
            # print("LOWER", candle[1], candle[2])
            ts = 0
        trend_state.append(ts)
        # print("TrendSTATE", ts, trend_state)

        # print("CANDLE", index, candle) # [t, o, h, l, c, v]
        # if index == 0:
        #     current = {
        #         "Time": candles[0][0],
        #         "Point": candles[0][2],
        #         "Inv": candles[0][3],
        #         "Direction": -1 if candles[0][2] < candles[0][3] else 1,
        #     }
        #     trendlines.append(current)
        #     trend_state.append(-1 if candles[0][2] < candles[0][3] else 1)
        #     pass
        # else:
        #     # Higher High in uptrend  (continuation)
        #     if candle[2] > current["Point"] and current["Direction"] == 1:
        #         current = {
        #             "Time": candle[0],
        #             "Point": candle[2],
        #             "Inv": candle[3],
        #             "Direction": 1,
        #         }
        #         ts = 1

                
        #     # Higher High in downtrend  (new trend)
        #     if (candle[2] > current["Inv"]  and current["Direction"] == 0):
        #         current = {
        #             "Time": candle[0],
        #             "Point": candle[2],
        #             "Inv": candle[3],
        #             "Direction": 0 if candle[2] < candle[3] else 1,
        #         }
        #         ts = 1
                
        #     # Lower Low in uptrend (new trend)
        #     if (candle[3] < current["Inv"]  and current["Direction"] == 1):
        #         current = {
        #             "Time": candle[0],
        #             "Point": candle[3],
        #             "Inv": candle[2],
        #             "Direction": 0 if candle[2] < candle[3] else 1,
        #         }
        #         ts = 0

                
                
        #     # Lower Low in downtrend  (continuation)
        #     if candle[3] < current["Point"] and current["Direction"] == 0:
        #         current = {
        #             "Time": candle[0],
        #             "Point": candle[3],
        #             "Inv": candle[2],
        #             "Direction": 0 if candle[2] < candle[3] else 1,
        #         }
        #         ts = 0


        #     # Janky fix to get over skipping trends that have the same unique time for start and end
        #     # if current["StartTime"] == current["EndTime"]:
        #     #     current["StartTime"] -= 1
        #     trendlines.append(current)
        #     # print("CANDLE LENGTH", len(current))
        #     # print("TREND", ts)
        #     trend_state.append(ts)
            
        # # print("Trendline", current, "\n")


    return trend_state


# def mean_normalize(data):
#     mean_vals = np.mean(data, axis=0)
#     min_vals = np.min(data, axis=0)
#     max_vals = np.max(data, axis=0)
#     return (data - mean_vals) / (max_vals - min_vals)

def mean_normalize(data, mean_vals, range_vals):
    return (data - mean_vals) / range_vals
def load_dataset():
    connection = mysql.connector.connect(
        host="localhost",
        user="root",
        password="1234567",
        database="markets"
    )

    cursor = connection.cursor()

    candles_query = "SELECT * FROM coinbase_ETHUSD_1 ORDER BY time"
    cursor.execute(candles_query)
    candles = cursor.fetchall()
    print(f"{len(candles)} Candles \n")
    # Limit to 1000 candles
    candles = candles[:100000]

    # Assuming make_trendlines is a function defined elsewhere that you're using
    trends = make_trendlines(candles)

    cursor.close()
    connection.close()

    # Split the data into 80:20 training and test
    split_index = math.floor(len(candles) * 0.8)
    candles_training = candles[:split_index]
    candles_test = candles[split_index:]

    trends_training = trends[:split_index]
    trends_test = trends[split_index:]

    # Extract the relevant features from candles
    candles_training = np.array([[candle[1], candle[2], candle[3], candle[4]] for candle in candles_training])
    candles_test = np.array([[candle[1], candle[2], candle[3], candle[4]] for candle in candles_test])

    # Calculate mean and range for mean normalization using training data
    mean_vals = np.mean(candles_training, axis=0)
    range_vals = np.max(candles_training, axis=0) - np.min(candles_training, axis=0)

    # Normalize the candles data
    candles_training_normalized = mean_normalize(candles_training, mean_vals, range_vals)
    candles_test_normalized = mean_normalize(candles_test, mean_vals, range_vals)

    # Convert normalized candles data to tensors
    X_train_tensor = torch.tensor(candles_training_normalized, dtype=torch.float)
    X_test_tensor = torch.tensor(candles_test_normalized, dtype=torch.float)

    # Convert trends data to tensors
    y_train_tensor = torch.tensor(trends_training, dtype=torch.float)
    y_test_tensor = torch.tensor(trends_test, dtype=torch.float)

    return X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor

X_train, y_train, X_test, y_test = load_dataset()
X_train, y_train = X_train.to(device), y_train.to(device)
X_test, y_test = X_test.to(device), y_test.to(device)


# Train Model







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

# Evaluation function for accuracy
def calculate_accuracy(y_true, y_pred_logits):
    y_pred = torch.round(torch.sigmoid(y_pred_logits))  # Convert logits to probabilities and then to binary labels
    # print(f"PREDICTIONS {y_true} \n {y_pred}\n")
    correct = (y_pred == y_true).float()  # Compare predicted labels with true labels
    accuracy = correct.sum() / len(correct)  # Calculate accuracy
    return accuracy.item() * 100  # Return accuracy as a percentage

# Initialize the model
model = BinaryClassifier().to(device)

# Loss function
loss_fn = nn.BCEWithLogitsLoss()

# Optimizer (using Adam)
optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=5e-5)

train_accuracies = []
test_accuracies = []
loss_array = []

# Turn on interactive mode
# plt.ion()

# Initialize your figure
# fig, ax = plt.subplots()
# ax.set_xlabel('Epoch')
# ax.set_ylabel('Accuracy (%)')
# ax.set_title('Training and Test Accuracy')
# fig.patch.set_facecolor('#212f3d')  # Changing the figure background color

# # Set the axes face color
# ax.set_facecolor('#212f3d')
# # Lines for training and test accuracies
# line1, = ax.plot([], [], label='Training Accuracy')
# line2, = ax.plot([], [], label='Test Accuracy')
# ax.legend()

def run_validation():
    model.eval()

# Training loop
num_epochs = 10001
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

    # # Update the data for the plot
    # line1.set_ydata(train_accuracies)
    # line1.set_xdata(range(epoch+1))
    # line2.set_ydata(test_accuracies)
    # line2.set_xdata(range(epoch+1))
    # ax.relim()  # Recalculate limits
    # ax.autoscale_view()  # Rescale the axis

    # # Redraw the figure
    # fig.canvas.draw()
    # plt.pause(0.001)  # Pause for interval seconds

# Turn off interactive mode
# plt.ioff()

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