from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
import pandas as pd
import torch
from torch import nn
import matplotlib.pyplot as plt



# -----------------------------------------------------------------------

class Circle_Model_0(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_1 = nn.Linear(in_features=2, out_features=5)
        self.layer_2 = nn.Linear(in_features=5, out_features=1)

    def forward(self, x):
        return self.layer_2(self.layer_1(x))
n_samples = 1000

X, y = make_circles(n_samples, noise=0.03, random_state=42)

# print(f"First 5 X features:\n{X[:5]}")
# print(f"First 5 y features:\n{y[:5]}\n")

circles = pd.DataFrame({"X1": X[:, 1], "X2": X[:, 1], "label": y})
# print(f"\nCicles_Head\n{circles.head(10)}\nCircles_Label\n{circles.label.value_counts()}\nShape:\n X: {X.shape}\n y:{y.shape}")

X_sample = X[0]
y_sample = y[0]
# print(f"Values for one sample of X: {X_sample} and the same for y: {y_sample}")
# print(f"Shapes for one sample of X: {X_sample.shape} and the same for y: {y_sample.shape}")

X = torch.from_numpy(X).type(torch.float)
y = torch.from_numpy(y).type(torch.float)

X[:5], y[:5]
# print(f"\n Tensor first 5 samples. \n X: {X[:5]}\n y: {y[:5]} \n")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# print(f"X_train: {len(X_train)} y_train: {len(y_train)} X_test: {len(X_test)} y_test: {len(y_test)}")

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# print(f"\nDevice: {device} \n")

model_0 = Circle_Model_0().to(device)
model_0 = nn.Sequential(
    nn.Linear(in_features=2, out_features=5),
    nn.Linear(in_features=5, out_features=1)
).to(device)
print(model_0)

# Make predictions with the model
untrained_preds = model_0(X_test.to(device))
print(f"Length of predictions: {len(untrained_preds)}, Shape: {untrained_preds.shape}")
print(f"Length of test samples: {len(y_test)}, Shape: {y_test.shape}")
print(f"\nFirst 10 predictions:\n{untrained_preds[:10]}")
print(f"\nFirst 10 test labels:\n{y_test[:10]}")

# Loss Function
loss_fn = nn.BCEWithLogitsLoss()

# Optimizer
optimizer = torch.optim.SGD(params=model_0.parameters(), lr=0.1)

# Calculate accuracy (A classification metric) 
def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct / len(y_pred)) * 100
    return acc

# 
# Use sigmoid on model logits
y_logits = model_0(X_test.to(device))
y_pred_probs = torch.sigmoid(y_logits)
# Find the predicted labels (round the prediction probabilities)
y_preds = torch.round(y_pred_probs)
# In Full
y_pred_labels = torch.round(torch.sigmoid(model_0(X_test.to(device))))
# Check for equality
print(torch.eq(y_preds.squeeze(), y_pred_labels.squeeze()))
# Get rid of extra dimension
y_preds.squeeze()
print(y_test[:5])

# -----------------------------------------------------------------------


torch.manual_seed(42)

epochs = 100

X_train, y_train = X_train.to(device), y_train.to(device)
X_test, y_test = X_test.to(device), y_test.to(device)

# Build training and evaluuation loop
for epoch in range(epochs):

    # Training:
    model_0.train()

    # 1. Forward Pass(Model outputs raw logits)
    y_logits = model_0(X_train).squeeze()
    y_pred = torch.round(torch.sigmoid(y_logits))
    # 2. Calculate loss/accuracy
    loss = loss_fn(y_logits, y_train)
    acc = accuracy_fn(y_true=y_train, y_pred=y_pred)
    # 3. Optimizer zero grad
    optimizer.zero_grad()
    # 4. Loss backwards
    loss.backward()
    # 5. Optimizer step
    optimizer.step()

    # Testing:
    model_0.eval()
    with torch.inference_mode():
        # 1. Forward pass
        test_logits = model_0(X_test).squeeze()
        test_pred = torch.round(torch.sigmoid(test_logits))
        # 2. Calculate loss/accuracy
        test_loss = loss_fn(test_logits, y_test)
        test_acc = accuracy_fn(y_true=y_test, y_pred=test_pred)

        # Print Epoch
        if epoch % 10 == 0:
            print(f"Epoch: {epoch} | Loss: {loss:.5f}, Accuracy: {acc:.2f}% | Test loss: {test_loss:.5f}, Test acc: {test_acc:.2f}%")


# -----------------------------------------------------------------------
# plt.scatter(x=X[:, 0], 
#             y=X[:, 1], 
#             c=y, 
#             cmap=plt.cm.RdYlBu);

# plt.show()