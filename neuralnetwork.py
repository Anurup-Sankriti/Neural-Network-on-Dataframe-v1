import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import TensorDataset
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split



class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralNetwork, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = self.layer1(x)
        out = self.relu(out)
        out = self.layer2(out)
        return out.squeeze(1)







df = pd.read_csv('emails.csv')
df.columns = [c.replace(' ', '_') for c in df.columns]
#print(df.shape)
print(df.head())
#print(df.columns)


x = df.drop('Email_No.', axis=1)
X= x.drop('Prediction', axis=1)
#print(X)
Y = df['Prediction']
# print(Y)

np.random.seed(42)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
#print( X_train.shape[1]*X_train.shape[0])



input_size = X_train.shape[1]
hidden_size = 64
output_size = 1  # Assuming it's a regression problem
model = NeuralNetwork(input_size, hidden_size, output_size)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 100
batch_size = 32

X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32)

dataset = torch.utils.data.TensorDataset(X_train_tensor,y_train_tensor)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

#print(dataset)

for epoch in range(num_epochs):
    for batch_X, batch_y in dataloader:
        # Forward pass
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)


        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Print the progress
    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
with torch.no_grad():
    predicted = model(X_test_tensor)
