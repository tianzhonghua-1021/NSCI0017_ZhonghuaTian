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
            nn.Linear(15, 64), # 2 Input features: T and P
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)  # 1 Output: predicted theta
        )
        
    def forward(self, x):
        return self.net(x)

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
    
    # Forward pass
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    
    # Backward pass and optimization
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
plt.figure(figsize=(12, 5))

# Plot Learning Curve (Loss over epochs)
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Training Loss', color='blue')
plt.title('Learning Curve (Loss)')
plt.xlabel('Epochs')
plt.ylabel('MSE Loss')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()

# Plot Parity Plot (Predicted vs True)
plt.subplot(1, 2, 2)
plt.scatter(y_test, y_pred, alpha=0.5, color='green', label=f'R² = {r2:.3f}')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Ideal Line')
plt.title('Prediction Accuracy (T, P Features Only)')
plt.xlabel('DFT Calculated Theta')
plt.ylabel('MLP Predicted Theta')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()

plt.tight_layout()
plt.savefig('mlp_result.png')
plt.show()