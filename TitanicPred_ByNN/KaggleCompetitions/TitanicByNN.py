import numpy
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader,Dataset
import tqdm
# 构建训练集和测试集
datap = pd.read_csv("./titanic/train.csv")
datax = pd.read_csv("./titanic/test.csv")
class TiTanicData(Dataset):
    def __init__(self,datap):
        #读取数据并初步洗去没用的列
        datap.loc[datap.Sex == 'male','Sex'] =1
        datap.loc[datap.Sex == 'female', 'Sex'] = 0
        datap.loc[pd.isnull(datap.Age),'Age']=-1
        datap = datap.drop(labels=[
            'PassengerId','Name','Ticket','Cabin','Embarked'
        ],axis=1)
        # 定义训练集的输入参数
        self.x_data =torch.from_numpy(
            np.array(
                datap.iloc[:, 1:7].astype(np.float32)
            )
        )
        # 定义训练集的输出参数
        self.y_data = torch.from_numpy(
            np.array(datap['Survived']).astype(np.float32))
    def __getitem__(self, item):
        return self.x_data[item],self.y_data[item]

    def __len__(self):
        return len(datap)

train_dataset = TiTanicData(datap)
train_dataloader = DataLoader(
    dataset=train_dataset, batch_size=8, shuffle=True)
# 定义模型
class TiTanic(torch.nn.Module):
    def __init__(self):
        super(TiTanic, self).__init__()
        self.l1 = torch.nn.Linear(6,122)
        self.l2 = torch.nn.Dropout(p=0.5)
        self.l3 = torch.nn.Linear(122, 32)
        self.l4 = torch.nn.Linear(32, 2)
        self.l5 = torch.nn.Linear(2,1)
    def forward(self,x):
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        x = torch.sigmoid(self.l5(x))
        return x
# 构建模型参数
model = TiTanic()
criterion = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(),lr=0.01,momentum=0.1)
# 训练模型
for epoch in range(20):
    train_loss=[]
    for idx,data in tqdm.tqdm(enumerate(train_dataloader,0)):
        input ,label = data
        y_pred = model(input)
        y_pred = y_pred.squeeze(-1)
        loss = criterion(y_pred,label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss.append(loss.item())
        if idx % 10 == 0:
            print(epoch, idx, np.mean(train_loss))
    if epoch % 5 == 0:
        print(train_loss)
# 验证

    dev_mean_loss = 0
    with torch.no_grad():
        correct = 0
        total = 0
        for data in train_dataloader:
            inputs,labels = data
            outputs = model(inputs).squeeze(-1)
            dev_loss = criterion(outputs, labels)
            dev_mean_loss += dev_loss.item()
            total += labels.size(0)
            correct += (np.round(outputs) == labels).sum().item()
        print('Accuracy on train set:%d %%' % (100*correct / total))
        print(total)
# 预测模型

datax1 = datax.drop(labels=[
    'PassengerId','Name','Ticket','Cabin','Embarked'
],axis=1)
datax1.loc[datax1.Sex == 'male','Sex'] =1
datax1.loc[datax1.Sex == 'female', 'Sex'] = 0
datax1.loc[pd.isnull(datax1.Age),'Age']=-1
datax1 = torch.from_numpy(
            np.array(
                datax1.astype(np.float32)
            )
        )
with torch.no_grad():
    y_pred1 = model(datax1)
    y = []
    y_pred1 = y_pred1.detach().numpy().squeeze()
    for i in y_pred1:
        y.append(np.round(i)) #四舍五入，y>=0.5认为存活，否则视为死亡
    #预测结果保存为csv文件
    output = pd.DataFrame({'Survived_pre': y})
    data1 = pd.read_csv("./titanic/gender_submission.csv")
    data1['Survived_pre'] = output
    data1.to_csv('./titanic/gender_submission.csv',index=False,sep=',')
predict = pd.read_csv("./titanic/gender_submission.csv")
predict.loc[predict.Survived == predict.Survived_pre,'Survived_pre'] = 1
count = 0
for i in predict.Survived_pre:
    if i == 1.0:
        count+=1
print(count / 417)