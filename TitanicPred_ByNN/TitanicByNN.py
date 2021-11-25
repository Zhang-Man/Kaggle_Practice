# 导入相关包
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import tqdm
import seaborn as sns
# ------------------------------------------------------------------------- #
"""
    EDA
    加载数据
    清洗数据
    构建数据集    
    构建相关图像
"""
# 数据分析
test_df = pd.read_csv('./titanic/test.csv')
train_df = pd.read_csv('./titanic/train.csv')
def fills_good(data, main_column, function, group_columns):
    data[main_column] = data[main_column].fillna(data.groupby(group_columns)[main_column].transform(function))
    return data[main_column]
# 加载数据
test_target = pd.read_csv('./titanic/gender_submission.csv')
test_concat = pd.merge(test_df, test_target, on='PassengerId')
main_df = pd.concat([train_df,test_concat], ignore_index=True,)
main_df = pd.get_dummies(main_df, columns=['Sex'],drop_first=True)
main_df = main_df.rename(columns={'Sex_male':'Sex'})
main_df.head()
print('-'*40)
main_df.info()
print('-'*40)

# 查看数据的相关信息
main_df.describe()

# 寻找缺失的值
sns.set(palette='BuGn_r')
fig = plt.figure(figsize = (12,9))
sns.histplot(main_df["Age"], kde=True)
plt.title('Age hist')
plt.show()

# 绘制相关性图像
sns.pairplot(main_df[['Age', 'Survived', 'Pclass', 'SibSp', 'Parch', 'Sex']],palette='CMRmap_r')
plt.show()
# 填补空缺的值
main_df['Age'] = fills_good(main_df, 'Age', 'median', ['Pclass', 'SibSp', 'Parch', 'Embarked', 'Sex'])
main_df['Age'] = main_df['Age'].fillna(main_df['Age'].median())
main_df.info()

fig = plt.figure(figsize = (12,9))
sns.histplot(main_df["Age"], kde=True, palette='BuPu_r')
plt.title('Age hist after filling')
plt.show()

fig, ax = plt.subplots(figsize=(10,10))
colors = sns.color_palette("Accent")
ax.pie(main_df["Embarked"].value_counts(), labels=main_df["Embarked"].value_counts().index, autopct='%.0f%%', colors = colors)
ax.set_title('Survery responses')
plt.show()

main_df['Embarked'] = main_df['Embarked'].fillna('S')
main_df['Fare'] = main_df['Fare'].fillna(main_df['Fare'].median())
main_df = main_df.drop(labels='Cabin', axis=1)

# 数据分析
main_df.info()
print('-'*40)
main_df.describe()
print('-'*40)
main_df.describe(include=['O'])

main_df[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)

main_df[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)

main_df[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean().sort_values(by='Survived', ascending=False)

main_df[['SibSp', 'Survived']].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False)

main_df[['Parch', 'Survived']].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived', ascending=False)

fig = sns.FacetGrid(main_df,
                    col='Survived',
                    height=5,
                    palette='BuGn_r')
fig.map(sns.histplot, 'Age',kde=True ,bins=20)
plt.show()
grid = sns.FacetGrid(main_df, col='Survived', row='Sex',height=5)
grid.map(sns.histplot, 'Age', alpha=.5, bins=20, kde=True)
grid.add_legend()
plt.show()
grid = sns.FacetGrid(main_df, col='Survived', row='Pclass',height=5)
grid.map(sns.histplot, 'Age', alpha=.5, bins=20, kde=True)
grid.add_legend()
plt.show()
main_df = main_df.drop(['Name', 'Ticket'], axis=1)

main_df['Age_group'] = pd.cut(main_df['Age'], 5)
main_df[['Age_group', 'Survived']].groupby(['Age_group'], as_index=False).mean().sort_values(by='Age_group', ascending=True)

main_df.loc[ main_df['Age'] <= 16, 'Age'] = 0
main_df.loc[(main_df['Age'] > 16) & (main_df['Age'] <= 32), 'Age'] = 1
main_df.loc[(main_df['Age'] > 32) & (main_df['Age'] <= 48), 'Age'] = 2
main_df.loc[(main_df['Age'] > 48) & (main_df['Age'] <= 64), 'Age'] = 3
main_df.loc[ main_df['Age'] > 64, 'Age'] = 4
main_df.Age = main_df.Age.astype('int')
main_df = main_df.drop('Age_group', axis=1)
main_df.head()
main_df['Fare_group'] = pd.cut(main_df['Fare'], 5)
main_df[['Fare_group', 'Survived']].groupby(['Fare_group'], as_index=False).mean().sort_values(by='Fare_group', ascending=True)
main_df.loc[ main_df['Fare'] <= 102, 'Fare'] = 0
main_df.loc[(main_df['Fare'] > 102) & (main_df['Fare'] <= 204), 'Fare'] = 1
main_df.loc[(main_df['Fare'] > 204) & (main_df['Fare'] <= 307), 'Fare'] = 2
main_df.loc[(main_df['Fare'] > 307) & (main_df['Fare'] <= 409), 'Fare'] = 3
main_df.loc[ main_df['Fare'] > 409, 'Fare'] = 4
main_df.Age = main_df.Age.astype('int')
main_df = main_df.drop('Fare_group', axis=1)
main_df.head()
main_df_ohe = pd.get_dummies(main_df, columns=['Pclass', 'SibSp','Fare', 'Embarked', 'Age', ],drop_first=True)
test_df_ohe = main_df_ohe[main_df_ohe['PassengerId'].isin(test_df.PassengerId)]
train_df_ohe = main_df_ohe[main_df_ohe['PassengerId'].isin(train_df.PassengerId)]
train_target = train_df_ohe['Survived']
train_df_ohe = train_df_ohe.drop(['Survived', 'PassengerId'], axis = 1)
test_df_ohe = test_df_ohe.drop(['Survived', 'PassengerId'], axis = 1)

train_target.shape
print("-"*40)
train_df_ohe.shape
print("-"*40)
test_df_ohe.shape

test_target = test_target.drop('PassengerId',axis=1)
# 加载数据
datap = pd.read_csv("./titanic/train.csv")
datax = pd.read_csv("./titanic/test.csv")
# 这里可以对数据集的数据进行分析
class TiTanicData(Dataset):
    def __init__(self, datap):
        # 读取数据并
        # 初步洗去没用的列
        datap.loc[datap.Sex == 'male', 'Sex'] = 1
        datap.loc[datap.Sex == 'female', 'Sex'] = 0
        datap.loc[pd.isnull(datap.Age), 'Age'] = -1
        datap = datap.drop(labels=[
            'PassengerId', 'Name', 'Ticket', 'Cabin', 'Embarked'
        ], axis=1)
        # 定义训练集的输入参数
        self.x_data = torch.from_numpy(
            np.array(
                datap.iloc[:, 1:7].astype(np.float32)
            )
        )
        # 定义训练集的输出参数
        self.y_data = torch.from_numpy(
            np.array(datap['Survived']).astype(np.float32))

    def __getitem__(self, item):
        return self.x_data[item], self.y_data[item]

    def __len__(self):
        return len(datap)


# 构建数据集
train_dataset = TiTanicData(datap)
train_dataloader = DataLoader(
    dataset=train_dataset, batch_size=8, shuffle=True)
# ------------------------------------------------------------------------- #
"""
    模型部分
    选择模型
    搭建模型骨架
    选择模型的优化器和损失函数
    调整模型参数
"""


# ------------------------------------------------------------------------- #
# 定义模型
class TiTanic(torch.nn.Module):
    def __init__(self):
        super(TiTanic, self).__init__()
        self.l1 = torch.nn.Linear(6, 122)
        self.l2 = torch.nn.Dropout(p=0.5)
        self.l3 = torch.nn.Linear(122, 32)
        self.l4 = torch.nn.Linear(32, 2)
        self.l5 = torch.nn.Linear(2, 1)

    def forward(self, x):
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        x = torch.sigmoid(self.l5(x))
        return x


# 构建模型参数
model = TiTanic()
criterion = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.1)
# ------------------------------------------------------------------------- #
"""
    训练部分
    调整训练的相关参数 批和量
    保存训练的相关参数 Loss or Acc
    构建相关图像
"""
# ------------------------------------------------------------------------- #
# 训练模型
for epoch in tqdm.tqdm(range(20)):
    train_loss = []
    for idx, data in enumerate(train_dataloader, 0):
        input, label = data
        y_pred = model(input)
        y_pred = y_pred.squeeze(-1)
        loss = criterion(y_pred, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss.append(loss.item())
        if idx % 10 == 0:
            print(epoch, idx, np.mean(train_loss))
    if epoch % 5 == 0:
        print(train_loss)
plt.plot(train_loss)
plt.show()
# ------------------------------------------------------------------------- #

"""
    保存结果模型
    对于测试集进行测试
    保存测试集结果
    构建相关图像
    保存结果
"""
# ------------------------------------------------------------------------- #
# 验证

dev_mean_loss = 0
with torch.no_grad():
    correct = 0
    total = 0
    for data in train_dataloader:
        inputs, labels = data
        outputs = model(inputs).squeeze(-1)
        dev_loss = criterion(outputs, labels)
        dev_mean_loss += dev_loss.item()
        total += labels.size(0)
        correct += (np.round(outputs) == labels).sum().item()
    print('Accuracy on train set:%d %%' % (100 * correct / total))
    print(total)
# 预测模型

datax1 = datax.drop(labels=[
    'PassengerId', 'Name', 'Ticket', 'Cabin', 'Embarked'
], axis=1)
datax1.loc[datax1.Sex == 'male', 'Sex'] = 1
datax1.loc[datax1.Sex == 'female', 'Sex'] = 0
datax1.loc[pd.isnull(datax1.Age), 'Age'] = -1
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
        y.append(np.round(i))
    # 四舍五入，y>=0.5认为存活，否则视为死亡
    # 预测结果保存为csv文件
    output = pd.DataFrame({'Survived_pre': y})
    data1 = pd.read_csv("./titanic/gender_submission.csv")
    data1['Survived_pre'] = output
    data1.to_csv('./titanic/gender_submission.csv', index=False, sep=',')
predict = pd.read_csv("./titanic/gender_submission.csv")
predict.loc[predict.Survived == predict.Survived_pre, 'Survived_pre'] = 1
count = 0
for i in predict.Survived_pre:
    if i == 1.0:
        count += 1
print(count / 417)
