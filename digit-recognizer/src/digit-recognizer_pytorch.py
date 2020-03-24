# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# %%
train = pd.read_csv('../data/train.csv').astype(float)
test = pd.read_csv('../data/test.csv').astype(float)

# %%
train_y = train.loc[:, 'label'].astype(int)
train_x = train.drop('label', axis=1)

# %%
train_x = train_x / 255.0
test = test / 255.0

train_x = torch.FloatTensor(np.array(train_x)).to(device)
train_y = torch.IntTensor(np.array(train_y)).to(device)
test = torch.FloatTensor(np.array(test)).to(device)


# %%
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(1, 32, 5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        self.linear = nn.Linear(64, 10)

    def forward(self, x):
        x = self.layers(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x


# %%
EPOCH_NUM = 50
BATCH_SIZE = 1024

train = data.TensorDataset(train_x, train_y)
train_loader = data.DataLoader(train, batch_size=BATCH_SIZE, shuffle=True)

net = Net().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.01)


# %%
total_loss_log = []
for epoch in range(1, EPOCH_NUM + 1):
    total_loss = 0
    for x, y in train_loader:
        optimizer.zero_grad()

        output = net(x)
        loss = criterion(output, y)
        total_loss += loss.item()
        loss.backward()
        optimizer.step()

    total_loss_log.append(total_loss)

    print('Epoch: %d | Total Loss: %d' % (epoch, total_loss))

plt.plot([i for i in range(EPOCH_NUM)], total_loss_log)


# %%
net.eval()
result = torch.max(net(test), 1)[1]

# %%
submission = pd.DataFrame()
submission['ImageId'] = [i for i in range(1, len(result.numpy()))]
submission['Label'] = result.numpy()
submission.to_csv('submission.csv', index=False)


# %%
