import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Placeholder for dataset loading
X_train = torch.randn(1000, 10)  # Example feature set
y_train = torch.randint(0, 2, (1000,))  # Example labels (binary classification)

# Create DataLoader
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Define the Model
class SpliceJunctionModel(nn.Module):
    def __init__(self):
        super(SpliceJunctionModel, self).__init__()
        self.fc1 = nn.Linear(10, 50)
        self.fc2 = nn.Linear(50, 2)  # Binary classification output

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = SpliceJunctionModel()


# Train the model 
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    for batch in train_loader:
        inputs, labels = batch
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}")

# Save the Model
model_path = "./splam/test/test_script.pt"
torch.save(model.state_dict(), model_path)
print(f"Model saved to {model_path}")