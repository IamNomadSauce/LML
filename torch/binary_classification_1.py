from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
import pandas as pd
import torch
from torch import nn
import matplotlib.pyplot as plt
import math
import mysql.connector
import numpy as np

# -----------------------------------------------------------------------

class Circle_Model_0(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_1 = nn.Linear(in_features=6, out_features=10)
        self.layer_2 = nn.Linear(in_features=10, out_features=1)

    def forward(self, x):
        return self.layer_2(self.layer_1(x))

class Circle_Model_1(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_1 = nn.Linear(in_features=6, out_features=18)
        self.layer_2 = nn.Linear(in_features=18, out_features=18)
        self.layer_3 = nn.Linear(in_features=18, out_features=1)
        self.relu = nn.ReLU() # <- add in ReLU activation function
        # Can also put sigmoid in the model 
        # This would mean you don't need to use it on the predictions
        # self.sigmoid = nn.Sigmoid()

    def forward(self, x):
      # Intersperse the ReLU activation function between layers
       return self.layer_3(self.relu(self.layer_2(self.relu(self.layer_1(x)))))

# # Calculate accuracy (A classification metric) 
def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct / len(y_pred)) * 100
    return acc

def sigmoid(x):
    return 1 / (1 + torch.exp(-x))



# -----------------------------------------------------------------------

# n_samples = 1000

# X, y = make_circles(n_samples, noise=0.03, random_state=42)

# # print(f"First 5 X features:\n{X[:5]}")
# # print(f"First 5 y features:\n{y[:5]}\n")

# circles = pd.DataFrame({"X1": X[:, 1], "X2": X[:, 1], "label": y})
# # print(f"\nCicles_Head\n{circles.head(10)}\nCircles_Label\n{circles.label.value_counts()}\nShape:\n X: {X.shape}\n y:{y.shape}")

# X_sample = X[0]
# y_sample = y[0]
# # print(f"Values for one sample of X: {X_sample} and the same for y: {y_sample}")
# # print(f"Shapes for one sample of X: {X_sample.shape} and the same for y: {y_sample.shape}")

# X = torch.from_numpy(X).type(torch.float)
# y = torch.from_numpy(y).type(torch.float)

# X[:5], y[:5]
# # print(f"\n Tensor first 5 samples. \n X: {X[:5]}\n y: {y[:5]} \n")

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # print(f"X_train: {len(X_train)} y_train: {len(y_train)} X_test: {len(X_test)} y_test: {len(y_test)}")

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"\nDevice: {device} \n")

# model_0 = Circle_Model_0().to(device)
# model_0 = nn.Sequential(
#     nn.Linear(in_features=2, out_features=5),
#     nn.Linear(in_features=5, out_features=1)
# ).to(device)
# print(model_0)

# # Make predictions with the model
# untrained_preds = model_0(X_test.to(device))
# print(f"Length of predictions: {len(untrained_preds)}, Shape: {untrained_preds.shape}")
# print(f"Length of test samples: {len(y_test)}, Shape: {y_test.shape}")
# print(f"\nFirst 10 predictions:\n{untrained_preds[:10]}")
# print(f"\nFirst 10 test labels:\n{y_test[:10]}")

# # Loss Function
# loss_fn = nn.BCEWithLogitsLoss()

# # Optimizer
# optimizer = torch.optim.SGD(params=model_0.parameters(), lr=0.1)



# # 
# # Use sigmoid on model logits
# y_logits = model_0(X_test.to(device))
# y_pred_probs = torch.sigmoid(y_logits)
# # Find the predicted labels (round the prediction probabilities)
# y_preds = torch.round(y_pred_probs)
# # In Full
# y_pred_labels = torch.round(torch.sigmoid(model_0(X_test.to(device))))
# # Check for equality
# print(torch.eq(y_preds.squeeze(), y_pred_labels.squeeze()))
# # Get rid of extra dimension
# y_preds.squeeze()
# print(y_test[:5])

# -----------------------------------------------------------------------

# epochs = 100

# X_train, y_train = X_train.to(device), y_train.to(device)
# X_test, y_test = X_test.to(device), y_test.to(device)

# Build training and evaluuation loop
# for epoch in range(epochs):

#     # Training:
#     model_0.train()

#     # 1. Forward Pass(Model outputs raw logits)
#     y_logits = model_0(X_train).squeeze()
#     y_pred = torch.round(torch.sigmoid(y_logits))
#     # 2. Calculate loss/accuracy
#     loss = loss_fn(y_logits, y_train)
#     acc = accuracy_fn(y_true=y_train, y_pred=y_pred)
#     # 3. Optimizer zero grad
#     optimizer.zero_grad()
#     # 4. Loss backwards
#     loss.backward()
#     # 5. Optimizer step
#     optimizer.step()

#     # Testing:
#     model_0.eval()
#     with torch.inference_mode():
#         # 1. Forward pass
#         test_logits = model_0(X_test).squeeze()
#         test_pred = torch.round(torch.sigmoid(test_logits))
#         # 2. Calculate loss/accuracy
#         test_loss = loss_fn(test_logits, y_test)
#         test_acc = accuracy_fn(y_true=y_test, y_pred=test_pred)

#         # Print Epoch
#         if epoch % 10 == 0:
#             print(f"Epoch: {epoch} | Loss: {loss:.5f}, Accuracy: {acc:.2f}% | Test loss: {test_loss:.5f}, Test acc: {test_acc:.2f}%")


# -----------------------------------------------------------------------
# Adjusting model_0 to fit a straight line (using nn.Sequential)
# -----------------------------------------------------------------------

# weight = 0.7
# bias = 0.3
# start = 0
# end = 1
# step = 0.01

# # Create Data
# X_regression = torch.arange(start, end, step).unsqueeze(dim=1)
# y_regression = weight * X_regression + bias

# print("\n---------------------------------------\n")
# # Check the data
# print(len(X_regression), "\n", X_regression[:5], "\n", y_regression[:5])

# # Create train and test splits
# train_split = int(0.8 * len(X_regression))
# X_train_regression, y_train_regression = X_regression[:train_split], y_regression[:train_split]
# X_test_regression, y_test_regression = X_regression[train_split:], y_regression[train_split:]

# # Check the lengths of each split
# print(len(X_train_regression), len(y_train_regression), "\n", len(X_test_regression), len(y_test_regression))

# model_2 = nn.Sequential(
#     nn.Linear(in_features=1, out_features=10),
#     nn.Linear(in_features=10, out_features=10),
#     nn.Linear(in_features=10, out_features=1),
# )
# print(model_2)

# # Loss and Optimizer
# loss_fn = nn.L1Loss()
# optimizer = torch.optim.SGD(model_2.parameters(), lr=0.1)

# torch.manual_seed(42)

# epochs = 1000

# # Put data to target device
# X_train_regression, y_train_regression = X_train_regression.to(device), y_train_regression.to(device)
# X_test_regression, y_test_regression = X_test_regression.to(device), y_test_regression.to(device)
# model_2.cuda()

# y_pred = 0
# for epoch in range(epochs):
#     # Training
#     # Forward Pass
#     y_pred = model_2(X_train_regression)

#     # Calculate the loss (There is no accuuracty because this is a regression problem, not a classification problem)
#     loss = loss_fn(y_pred, y_train_regression)

#     # Optimizer zero grad
#     optimizer.zero_grad()

#     # Loss backwards
#     loss.backward()

#     # Optimizer step
#     optimizer.step()


#     # Testing
#     model_2.eval()
#     with torch.inference_mode():
#         # Forward pass
#         y_pred = model_2(X_test_regression)

#         # Update the plot1 here 

#         # Calculate the loss
#         test_loss = loss_fn(y_pred, y_test_regression)
    
#     # Plot the train data, test data, y_predictions here, updated on each epoch

#     if epoch % 100 == 0:
#         print(f"Epoch: {epoch} | Train loss: {loss:.5f}, Test loss: {test_loss:.5f}")

# -----------------------------------------------------------------------
# Non-linearity
# -----------------------------------------------------------------------
# n_samples = 1000
# epochs = 1000
# torch.manual_seed(42)
# X, y = make_circles(n_samples=1000, noise=0.03, random_state=42)

# X = torch.from_numpy(X).type(torch.float)
# y = torch.from_numpy(y).type(torch.float)
# plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdBu);
# # Split data into train and test sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# model_3 = Circle_Model_1().to(device)
# print(model_3)

# loss_fn = nn.BCEWithLogitsLoss()
# optimizer = torch.optim.SGD(model_3.parameters(), lr=0.1)


# X_train, y_train = X_train.to(device), y_train.to(device)
# X_test, y_test = X_test.to(device), y_test.to(device)

# for epoch in range(epochs):
#     # 1. Forward pass
#     y_logits = model_3(X_train).squeeze()
#     y_pred = torch.round(torch.sigmoid(y_logits)) # logits -> prediction probabilities -> prediction labels
    
#     # 2. Calculate loss and accuracy
#     loss = loss_fn(y_logits, y_train) # BCEWithLogitsLoss calculates loss using logits
#     acc = accuracy_fn(y_true=y_train, 
#                       y_pred=y_pred)
    
#     # 3. Optimizer zero grad
#     optimizer.zero_grad()

#     # 4. Loss backward
#     loss.backward()

#     # 5. Optimizer step
#     optimizer.step()

#     ### Testing
#     model_3.eval()
#     with torch.inference_mode():
#       # 1. Forward pass
#       test_logits = model_3(X_test).squeeze()
#       test_pred = torch.round(torch.sigmoid(test_logits)) # logits -> prediction probabilities -> prediction labels
#       # 2. Calcuate loss and accuracy
#       test_loss = loss_fn(test_logits, y_test)
#       test_acc = accuracy_fn(y_true=y_test, y_pred=test_pred)

#     # Print out what's happening
#     if epoch % 100 == 0:
#         print(f"Epoch: {epoch} | Loss: {loss:.5f}, Accuracy: {acc:.2f}% | Test Loss: {test_loss:.5f}, Test Accuracy: {test_acc:.2f}%")
# # -----------------------------------------------------------------------
            

# model_3.eval()
# with torch.inference_mode():
#     y_preds = torch.round(torch.sigmoid(model_3(X_test))).squeeze()


# print(f"{y_preds[:10], y[:10]}")


# ----------------------------------------------------------------------
# Attempt to use custom candlestick data (Does not work with circular models)
# ----------------------------------------------------------------------
# def make_trendlines(candles):
#     trendlines = []
#     ts = 1
#     trend_state = []

#     for index, candle in enumerate(candles):
#         # print("CANDLE", index, candle) # [t, o, h, l, c, v]
#         if index == 0:
#             current = {
#                 "Time": candles[0][0],
#                 "Point": candles[0][2],
#                 "Inv": candles[0][3],
#                 "Direction": -1 if candles[0][2] < candles[0][3] else 1,
#             }
#             trendlines.append(current)
#             trend_state.append(-1 if candles[0][2] < candles[0][3] else 1)
#             pass
#         else:
#             # Higher High in uptrend  (continuation)
#             if candle[2] > current["Point"] and current["Direction"] == 1:
#                 current = {
#                     "Time": candle[0],
#                     "Point": candle[2],
#                     "Inv": candle[3],
#                     "Direction": 1,
#                 }
#                 ts = 1

                
#             # Higher High in downtrend  (new trend)
#             if (candle[2] > current["Inv"]  and current["Direction"] == 0):
#                 current = {
#                     "Time": candle[0],
#                     "Point": candle[2],
#                     "Inv": candle[3],
#                     "Direction": 0 if candle[2] < candle[3] else 1,
#                 }
#                 ts = 1
                
#             # Lower Low in uptrend (new trend)
#             if (candle[3] < current["Inv"]  and current["Direction"] == 1):
#                 current = {
#                     "Time": candle[0],
#                     "Point": candle[3],
#                     "Inv": candle[2],
#                     "Direction": 0 if candle[2] < candle[3] else 1,
#                 }
#                 ts = 0

                
                
#             # Lower Low in downtrend  (continuation)
#             if candle[3] < current["Point"] and current["Direction"] == 0:
#                 current = {
#                     "Time": candle[0],
#                     "Point": candle[3],
#                     "Inv": candle[2],
#                     "Direction": 0 if candle[2] < candle[3] else 1,
#                 }
#                 ts = 0


#             # Janky fix to get over skipping trends that have the same unique time for start and end
#             # if current["StartTime"] == current["EndTime"]:
#             #     current["StartTime"] -= 1
#             trendlines.append(current)
#             # print("CANDLE LENGTH", len(current))
#             # print("TREND", ts)
#             trend_state.append(ts)
            
#         # print("Trendline", current, "\n")


#     return trend_state

# def load_dataset():
#     connection = mysql.connector.connect(
#         host="localhost",
#         user="root",
#         password="1234567",
#         database="markets"
#     )

#     cursor = connection.cursor()

#     candles_query = "SELECT * FROM coinbase_BTCUSD_1 ORDER BY time"
#     trends_query = "SELECT * FROM coinbase_BTCUSD_1_trendlines"

#     cursor.execute(candles_query)
#     candles = cursor.fetchall()
#     # cursor.execute(trends_query)
#     # trends = cursor.fetchall()
#     trends = make_trendlines(candles)

#     # print(f"TRENDS {trends}")


#     cursor.close()
#     connection.close()
#     # print(f"Candles length {len(candles)} \n", f"TRENDS LENGTH {len(trends)} \n" )
    
#     # Split the data into 90:10 training and validation
#     candles_training = candles[:math.floor(len(candles) * 0.9)]
#     candles_test = candles[:math.floor(len(candles) * 0.1)]

#     trends_training = trends[:math.floor(len(trends) * 0.9)]
#     trends_test = trends[:math.floor(len(trends) * 0.1)]


#     # print(f"SPLIT DATA \n {len(trends_training)} \n {len(trends_test)} \n")

#     # Create test data with the candles[:candle_index]

#     X = candles_training
#     y = trends_training

#     X_test = candles_test
#     y_test = trends_test

#     # print(f"TRAINING DATA \n {len(X)}")
    
#     return np.array(X), np.array(y), np.array(X_test), np.array(y_test)

# n_samples = 1000
# epochs = 1000
# torch.manual_seed(42)
# X, y, X_test, y_test = load_dataset()


# X = torch.from_numpy(X).type(torch.float)
# y = torch.from_numpy(y).type(torch.float)
# # plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdBu);
# # # Split data into train and test sets
# X, X_test, y, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# model_3 = Circle_Model_1().to(device)
# print(model_3)

# loss_fn = nn.BCEWithLogitsLoss()
# optimizer = torch.optim.SGD(model_3.parameters(), lr=0.1)


# X, y = X.to(device), y.to(device)
# X_test, y_test = X_test.to(device), y_test.to(device)

# for epoch in range(epochs):
#     # 1. Forward pass
#     y_logits = model_3(X).squeeze()
#     y_pred = torch.round(torch.sigmoid(y_logits)) # logits -> prediction probabilities -> prediction labels
    
#     # 2. Calculate loss and accuracy
#     loss = loss_fn(y_logits, y) # BCEWithLogitsLoss calculates loss using logits
#     acc = accuracy_fn(y_true=y, 
#                       y_pred=y_pred)
    
#     # 3. Optimizer zero grad
#     optimizer.zero_grad()

#     # 4. Loss backward
#     loss.backward()

#     # 5. Optimizer step
#     optimizer.step()

#     ### Testing
#     model_3.eval()
#     with torch.inference_mode():
#       # 1. Forward pass
#       test_logits = model_3(X_test).squeeze()
#       test_pred = torch.round(torch.sigmoid(test_logits)) # logits -> prediction probabilities -> prediction labels
#       # 2. Calcuate loss and accuracy
#       test_loss = loss_fn(test_logits, y_test)
#       test_acc = accuracy_fn(y_true=y_test, y_pred=test_pred)

#     # Print out what's happening
#     if epoch % 100 == 0:
#         print(f"Epoch: {epoch} | Loss: {loss:.5f}, Accuracy: {acc:.2f}% | Test Loss: {test_loss:.5f}, Test Accuracy: {test_acc:.2f}%")


# model_3.eval()
# with torch.inference_mode():
#     y_preds = torch.round(torch.sigmoid(model_3(X_test))).squeeze()


# print(f"{y_preds[:10], y[:10]}")
# ----------------------------------------------------------------------


# Plot the train data, test data, y_predictions after the loop
plt.figure(figsize=(10, 5))
# print(f"{X_train.size()} \n {y_train.size()} \n {X_train[0]} - {y_train[0]}")
# Plot for X_train_regression
# plt.scatter(y_preds.cpu().numpy(), y.cpu().numpy(), c='blue', label='Train Data', s=1)

# # Plot for X_test
# plt.scatter(X_test.cpu().numpy(), y_test.cpu().numpy(), c='red', label='Test Data', s=1)

# # Plot for model predictions on X_test_regression
# plt.scatter(X_test.cpu().numpy(), y_pred.cpu().numpy(), c='green', label='Predictions', s=1)

# plt.title('Data and Predictions')
# plt.legend()
# plt.show()