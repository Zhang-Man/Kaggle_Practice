import pickle

import pandas
import numpy as np
import tqdm
from torch.utils.data import Dataset
import torch
from torch.utils.data import DataLoader
import torch.optim as optim


class myDataSet(Dataset):
    def __init__(self, train):
        np.set_printoptions(suppress=True)
        self.zcdatas = pandas.read_table('./新数据/预处理后的正常数据.txt', delimiter=':', encoding='gbk')
        self.ycdatas = pandas.read_table('./新数据/预处理后的异常数据.txt', delimiter=':', encoding='gbk')
        self.train = train
        zcflag = int(len(self.zcdatas) * 0.8)
        ycflag = int(len(self.ycdatas) * 0.8)

        self.trainSet = np.vstack((np.append(self.zcdatas.iloc[0:zcflag, 2:], np.ones((zcflag, 1)), axis=1),
                                   np.append(self.ycdatas.iloc[0:ycflag, 2:], np.zeros((ycflag, 1)), axis=1)))

        tezc = np.append(self.zcdatas.iloc[zcflag:, 2:], np.ones((len(self.zcdatas.iloc[zcflag:, 2:]), 1)),axis=1)
        teyc = np.append(self.ycdatas.iloc[ycflag:, 2:], np.zeros((len(self.ycdatas.iloc[ycflag:, 2:]), 1)),axis=1)
        self.testSet = np.vstack((tezc, teyc))
        np.random.shuffle(self.trainSet)
        np.random.shuffle(self.testSet)


    def __getitem__(self, item):
        if self.train:
            return self.trainSet[item]
        else:
            return self.testSet[item]

    def __len__(self):
        if self.train:
            return len(self.trainSet)
        else:
            return len(self.testSet)


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.bn = torch.nn.BatchNorm2d(1,affine=True)
        self.l1 = torch.nn.Linear(23,64)
        # self.l2 = torch.nn.Linear(128, 32)
        self.l2 = torch.nn.Linear(64,1)

    def forward(self, x):
        x = x.unsqueeze(1).unsqueeze(1)
        out = self.bn(x)
        out = out.squeeze(1).squeeze(1)

        out1 = torch.sigmoid(out)
        x = self.l1(out1)
        x = self.l2(torch.relu(x))
        #print()
        return torch.sigmoid(x)

def feature_normalize(data):
    mu = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    return (data - mu) / std


traindataset = myDataSet(True)
testdataset = myDataSet(False)

train_loader = DataLoader(dataset=traindataset, batch_size=128, shuffle=True, drop_last=False)
test_loader = DataLoader(dataset=testdataset,batch_size=64,shuffle=True,drop_last=False)

model = Net()
criterion = torch.nn.BCELoss()  # 交叉熵
optimize = optim.Adam(model.parameters(),lr=0.001)


def train(epoch):
    running_loss = 0.0
    for batch_idx, data in tqdm.tqdm(enumerate(train_loader, 0)):
        input = data[:, 0:-1].float()
        target = data[:, -1].float()
        optimize.zero_grad()
        output = model(input)
        output = torch.where(torch.isnan(output), torch.full_like(output, 0), output)
        loss = criterion(output.squeeze(), target)
        running_loss += loss.item()
        loss.backward()
        optimize.step()
        if (batch_idx % 100 == 99):
            losstext.append(running_loss / 100)
            print('[%d , %d]  Loss:%.3f' % (epoch + 1, batch_idx + 1, running_loss / 100))  # 计算这300批数据的平均loss
            running_loss = 0.0
losstext = []
succtext = []
def test():
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            input = data[:, 0:-1].float()
            # input = feature_normalize(np.array(input))
            # input = torch.tensor(input,requires_grad=True)
            labels = data[:, -1].float()
            output = model(input)
            one = torch.full_like(output, 1)
            zero = torch.full_like(output, 0)
            output = torch.where(output.gt(torch.full_like(output, 0.5)), one, zero)
            total += labels.size(0)  # 每一批=64个，所以total迭代一次加64
            correct += (output.squeeze() == labels.float()).sum().item()
    succtext.append(100 * correct / total)
    print('Accuracy on test set:%d %%' % (100 * correct / total))


if __name__ == "__main__":
    for epoch in range(15):
        train(epoch)  # 封装起来，若要修改主干就很方便
        test()
    # print(losstext)
    # print(succtext)
    # fileloss = open('./loss.txt', 'wb')
    # pickle.dump(losstext, fileloss,0)
    # filesucc = open('./succ.txt', 'wb')
    # pickle.dump(succtext, filesucc,0)
    torch.save(model, './model.pkl')


