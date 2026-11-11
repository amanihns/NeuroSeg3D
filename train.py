import torch
import torch.nn as nn
import torch.optim as optim

# ==========================================
# STEP 1: Prepare Data
# ==========================================
# Creating dummy tensors for input features (X) and target values (y)
# X represents medical scan data, y represents the tumor mask
X = torch.randn(2, 4, 32, 32, 32)  # Batch=2, Channels=4, Depth=32, Height=32, Width=32
y = torch.randint(0, 3, (2, 1, 32, 32, 32)).float()

print("1. Data prepared successfully!")
print(f"   - Input shape (X): {X.shape}")
print(f"   - Target shape (y): {y.shape}")

# ==========================================
# STEP 2: Build the Model
# ==========================================
# Defining a simple 3D CNN model architecture as a starting baseline
class Simple3DCNN(nn.Module):
    def __init__(self):
        super(Simple3DCNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(4, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(16, 1, kernel_size=3, padding=1)
        )
        
    def forward(self, x):
        return self.conv(x)

model = Simple3DCNN()
print("2. Model built successfully!")

# ==========================================
# STEP 3: Define Loss Function and Optimizer
# ==========================================
# Using BCEWithLogitsLoss for binary/segmentation tasks and Adam optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

print("3. Loss function and Optimizer defined!")

# ==========================================
# STEP 4: Train the Model
# ==========================================
# Running the training loop for multiple epochs
epochs = 5
print("4. Starting training loop...")

for epoch in range(epochs):
    # 1. Forward pass: generate predictions
    y_pred = model(X)
    
    # 2. Compute the loss (error)
    loss = criterion(y_pred, y)
    
    # 3. Backward pass and update weights
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # Print progress for each epoch
    print(f"   - Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")

print("Training finished successfully!")