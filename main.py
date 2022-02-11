import torch
import torch.nn as nn
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import os
import warnings
from BiLSTM import BidirectionalLSTM
from sklearn.model_selection import train_test_split


def get_device():
    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'
    return device


def preprocess(X, Y):
    x_train, x_valid, y_train, y_valid = train_test_split(X, Y, test_size=0.5)
    x_train = torch.from_numpy(x_train)
    x_valid = torch.from_numpy(x_valid)
    y_train = torch.from_numpy(y_train)
    y_valid = torch.from_numpy(y_valid)
    train_ds = TensorDataset(x_train, y_train)
    valid_ds = TensorDataset(x_valid, y_valid)

    return train_ds, valid_ds


def get_data(train_ds, valid_ds, bs):
    return (
        DataLoader(train_ds, batch_size=bs, shuffle=True),
        DataLoader(valid_ds, batch_size=bs),
    )


def bacc(model, x, y):
    row_num, _ = x.shape
    prediction = []
    x = torch.tensor(x)
    for i in range(row_num):
        outputs = model(x[i].view(1, 1, len(x[i])))
        predicted = torch.argmax(outputs)
        prediction.append(predicted)
    prediction = np.array(prediction)
    return (np.mean(prediction[np.where(y == 0)] == y[np.where(y == 0)]) +
            np.mean(prediction[np.where(y == 1)] == y[np.where(y == 1)])) / 2


def loss_batch(model, loss_func, xb, yb, opt=None):
    xb = xb.view(len(xb), 1, -1)
    yb = yb.view(len(yb), 1, -1)
    loss = loss_func(model(xb), yb)
    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()
    return loss.item(), len(xb)


def train(epochs, model, loss_func, train_dl, valid_dl):
    opt = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
    for epoch in range(epochs):
        model.train()
        for xb, yb in train_dl:
            loss_batch(model, loss_func, xb, yb, opt)

        model.eval()
        with torch.no_grad():
            losses, nums = zip(
                *[loss_batch(model, loss_func, xb, yb) for xb, yb in valid_dl]
            )

        val_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)

        print(epoch, val_loss)


DATA_PATH = 'data_set/'
warnings.filterwarnings("ignore")
epoch = 2
batch_size = 5
loss_func = nn.MSELoss()
hidden_dim = 64
torch.set_default_tensor_type(torch.DoubleTensor)
for data_name in os.listdir(DATA_PATH):
    data_array = pd.read_table(os.path.join(DATA_PATH, data_name),
                               header=None,
                               low_memory=False,
                               index_col=0).transpose().values
    features = data_array[:, 1:].astype('float')  # 每行是一个sample
    labels = data_array[:, 0]
    for number, label in enumerate(list(set(labels))):
        labels[np.where(labels == label)] = number
    labels = labels.astype('float')
    row_num, col_num = np.array(features).shape
    model = BidirectionalLSTM(col_num, hidden_dim, len(set(labels)))
    x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.5)
    train_ds, valid_ds = preprocess(x_train, y_train)
    train_dl, valid_dl = get_data(train_ds, valid_ds, batch_size)
    train(epoch, model, loss_func, train_dl, valid_dl)
    print(bacc(model, x_test, y_test))
