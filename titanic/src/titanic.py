# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim
from torch.autograd import Variable
import torch.utils.data as data


# %%
train = pd.read_csv('../data/train.csv')
test_x = pd.read_csv('../data/test.csv')
test_y = pd.read_csv('../data/gender_submission.csv')
train


# %%
# 'Cabin"のデータはNanを0、それ以外を1に置き換える
def cabin_encode(data):
    data['Cabin'] = data['Cabin'].apply(lambda x: 0 if pd.isnull(x) else 1)
    return data


# カテゴリ変数をLabel Encoding
def labeling(data, cat_cols):
    for cat in cat_cols:
        cat_i = data.columns.get_loc(cat)
        labels, _ = pd.factorize(data.iloc[:, cat_i])
        for i in range(len(labels)):
            data.iat[i, cat_i] = labels[i]

    return data


def standard(data, num_cols):
    # 数値データのNanを0に置き換える
    data[num_cols] = data[num_cols].fillna(0)
    # 数値データを正規化
    scaler = StandardScaler()
    data[num_cols] = scaler.fit_transform(data[num_cols])

    return data


ans_col = ['Survived']
cat_cols = ['Pclass', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked']
num_cols = list(set(train.columns) - set(ans_col + cat_cols))
train = cabin_encode(train)
train = labeling(train, cat_cols)
train = standard(train, num_cols)

cat_cols = ['Pclass', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked']
num_cols = list(set(test_x.columns) - set(cat_cols))
test_x = cabin_encode(test_x)
test_x = labeling(test_x, cat_cols)
test_x = standard(test_x, num_cols)


# %%
# IDと名前はユニークなデータなので削除
# Ticketはダブリが少ないので削除
del_cols = ['PassengerId', 'Name', 'Ticket']

# トレーニングデータを分割
x_cols = list(set(train.columns) - set(ans_col + del_cols))
train_x = train.drop(del_cols + ans_col, axis=1)
train_y = train.loc[:, ans_col]

# テストデータの必要のないカラムを削除
test_x = test_x.drop(del_cols, axis=1)
test_y = test_y.loc[:, ans_col]

# カテゴリ変数と正解データをOne Hot Encoding
cat_cols = list(set(cat_cols) - set(del_cols))
train_x = pd.get_dummies(train_x, columns=cat_cols)
train_y = pd.get_dummies(train_y, columns=ans_col)
test_x = pd.get_dummies(test_x, columns=cat_cols)

# %%
# トレインデータ確認
train_x

# %%
# テストデータ確認
test_x

# %%
# トレインデータに'Embarked_-1'があったので列を削除
train_x = train_x.drop('Embarked_-1', axis=1)


# %%
class Net(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout_p):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.do1 = nn.Dropout2d(dropout_p)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.do2 = nn.Dropout2d(dropout_p)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = f.relu(self.do1(self.fc1(x)))
        x = f.relu(self.do2(self.fc2(x)))
        x = self.fc3(x)
        return x


def to_cuda(obj):
    if torch.cuda.is_available():
        return obj.cuda()
    return obj


# %%
HIDDEN_SIZE = 32
DROPOUT_P = 0.2
EPOCH_NUM = 500
BATCH_SIZE = 10
input_size = len(train_x.columns)
outpu_size = len(train_y.columns)

train_x = torch.FloatTensor(np.array(train_x))
train_x = to_cuda(train_x)
train_y = torch.FloatTensor(np.array(train_y))
train_y = to_cuda(train_y)
train = data.TensorDataset(train_x, train_y)
train_loader = data.DataLoader(train, batch_size=BATCH_SIZE, shuffle=True)

net = Net(input_size, HIDDEN_SIZE, outpu_size, DROPOUT_P)
net = to_cuda(net)
print(net)

criterion = nn.MSELoss()
optimizer = optim.SGD(net.parameters(), lr=0.01)


# %%
# 学習
totalloss_log = []
for e in range(EPOCH_NUM):
    total_loss = 0
    for x, y in train_loader:
        x, y = Variable(x), Variable(y)
        optimizer.zero_grad()
        output = net(x)

        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    totalloss_log.append(total_loss)

    if (e + 1) % 20 == 0:
        print('Epoch: %d | Total Loss: %d' % (e + 1, total_loss))

plt.plot([i for i in range(len(totalloss_log))], totalloss_log)


# %%
test_x = torch.FloatTensor(np.array(test_x))
test_x = to_cuda(test_x)
test_y = torch.IntTensor(np.array(test_y))
test_y = to_cuda(test_y)
test_x, test_y = Variable(test_x), Variable(test_y)


# %%
net.eval()
result = torch.max(net(test_x).data, 1)[1]


def calc_accuracy(result, target):
    sum_correct = 0
    for r, t in zip(result, target):
        if r == t:
            sum_correct += 1

    return sum_correct / len(result) * 100.0


accuracy = calc_accuracy(result.numpy(), test_y.squeeze().data.numpy())
print('{}%'.format(accuracy))
# %%
