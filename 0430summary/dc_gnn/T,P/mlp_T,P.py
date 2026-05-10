import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt

# 1. Load and preprocess data
data_path = '/home/ucaqzti/models/dataset_form_tda.csv'
df = pd.read_csv(data_path)

# Extract T and P as features, theta as target
# Note: We are deliberately ignoring the File ID (structure) to see how much T/P alone can explain
# X = df[['TDA_H0_max','TDA_H0_min','TDA_H0_mean','TDA_H0_std','TDA_H0_sum','TDA_H1_max','TDA_H1_min','TDA_H1_mean','TDA_H1_std','TDA_H1_sum','TDA_H2_max','TDA_H2_min','TDA_H2_mean','TDA_H2_std','TDA_H2_sum']].values
X = df[['Temperature (K)','Pressure (bar)']].values
y = df['theta'].values.reshape(-1, 1)

# Split dataset into training and testing sets (Random split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature Scaling: crucial for MLP convergence
scaler_X = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

# Convert to PyTorch Tensors
X_train_tensor = torch.FloatTensor(X_train_scaled)
y_train_tensor = torch.FloatTensor(y_train)
X_test_tensor = torch.FloatTensor(X_test_scaled)
y_test_tensor = torch.FloatTensor(y_test)

# 2. Define a simple MLP Architecture
class LangmuirMLP(nn.Module):
    def __init__(self):
        super(LangmuirMLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 64), # 2 Input features: T and P
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)  # 1 Output: predicted theta
        )
        
    def forward(self, x):
        return self.net(x)
# define visualization function
def plot_learning_curve(losses, save_path='mlp_learning_curve.png'):
    plt.figure(figsize=(5, 5), dpi=300)

    epochs = range(1, len(losses) + 1)

    plt.plot(epochs, losses,
             label='Training Loss',
             color='#1f77b4',
             alpha=0.3)

    if len(losses) > 5:
        smooth_losses = np.convolve(losses, np.ones(5)/5, mode='valid')
        plt.plot(range(5, len(losses)+1),
                 smooth_losses,
                 label='Smoothed Loss (Moving Avg)',
                 color='#d62728',
                 lw=2)

    plt.title('MLP Training Convergence', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss (MSE)', fontsize=12)

    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc='upper right')

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
def plot_prediction_scatter(y_true, y_pred, r2, rmse,
                           save_path='mlp_result.png'):

    plt.figure(figsize=(5, 5), dpi=300)

    plt.scatter(
        y_true,
        y_pred,
        s=30,
        alpha=0.6,
        edgecolors='k',
        linewidths=0.3,
        c='#1f77b4',
        label='Predictions'
    )

    line_min = min(y_true.min(), y_pred.min())
    line_max = max(y_true.max(), y_pred.max())

    plt.plot([line_min, line_max],
             [line_min, line_max],
             'r--',
             lw=2,
             label='Ideal')

    stats_text = f"$R^2 = {r2:.4f}$\n$RMSE = {rmse:.4f}$"

    plt.gca().text(
        0.05, 0.95,
        stats_text,
        transform=plt.gca().transAxes,
        fontsize=12,
        verticalalignment='top',
        bbox=dict(
            boxstyle='round,pad=0.5',
            facecolor='white',
            alpha=0.8,
            edgecolor='gray'
        )
    )

    plt.xlabel('Actual Theta', fontsize=12)
    plt.ylabel('Predicted Theta', fontsize=12)
    plt.title('MLP Prediction: Actual vs Predicted',
              fontsize=14,
              fontweight='bold')

    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(loc='lower right')

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
# Initialize model, loss function and optimizer
model = LangmuirMLP()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 3. Training Loop
epochs = 500
train_losses = []

print("Starting training of pure MLP model (TDA only)...")

for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()

    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)

    loss.backward()
    optimizer.step()

    train_losses.append(loss.item())

    if (epoch + 1) % 20 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.6f}')

# 4. Evaluation
model.eval()
with torch.no_grad():
    y_pred_tensor = model(X_test_tensor)
    y_pred = y_pred_tensor.numpy()

# Calculate metrics
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"\n--- Evaluation Results ---")
print(f"R2 Score: {r2:.4f}")
print(f"RMSE: {rmse:.4f}")

# 5. Visualization

plot_learning_curve(train_losses)

plot_prediction_scatter(
    y_test.flatten(),
    y_pred.flatten(),
    r2,
    rmse
)
