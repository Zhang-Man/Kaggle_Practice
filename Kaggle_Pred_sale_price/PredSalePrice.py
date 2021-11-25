# 导入相关包
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader,Dataset
# 要求您预测下个月每种产品和商店的总销售额
# 设置gpu
device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# ------------------------------------------------------- #

"""
预测什么？我们需要给出每个商店的某种商品在一个月内的销售量
    数据部分
    对于当前训练数据集，数据相关的维度如下 
    1. date 每个产品售卖的时间
    2. date_block_num 时间区块编号
        分析后得此维度数据没有差值，均处于0值
    3. shop_id 商店id
    4. items_od 商品id
    5. items _price 每个商品价格
    6. item_cnt_day 某天售出的数量
    对于测试集
    1. ID 主键
    2. shop_id 商店id
    3. item_id 商品id
    加载数据
    清洗数据
    构建数据集    
    构建相关图像
"""
item_cat = pd.read_csv("./data/competitive-data-science-predict-future-sales/item_categories.csv")
items = pd.read_csv("./data/competitive-data-science-predict-future-sales/items.csv")
sales_data = pd.read_csv("./data/competitive-data-science-predict-future-sales/sales_train.csv")
shops = pd.read_csv("./data/competitive-data-science-predict-future-sales/shops.csv")
test_data = pd.read_csv("./data/competitive-data-science-predict-future-sales/test.csv")
sample_submission = pd.read_csv("./data/competitive-data-science-predict-future-sales/sample_submission.csv")

# Exploratory Data Analysis
# 探索性数据分析
def basic_eda(df):
    print("----------TOP 5 RECORDS--------")
    print(df.head(5))
    print("----------INFO-----------------")
    print(df.info())
    print("----------Describe-------------")
    print(df.describe())
    print("----------列--------------")
    print(df.columns)
    print("----------数据类型-----------")
    print(df.dtypes)
    print("-------NaN总数----------")
    print(df.isnull().sum())
    print("-------空值总数-------------")
    print(df.isna().sum())
    print("-----数据的Shape-------------")
    print(df.shape)
# 对数据集做一下上述操作
print("=============================Sales Data=============================")
basic_eda(sales_data)
print("=============================Test data=============================")
basic_eda(test_data)
print("=============================Item Categories=============================")
basic_eda(item_cat)
print("=============================Items=============================")
basic_eda(items)
print("=============================Shops=============================")
basic_eda(shops)
print("=============================Sample Submission=============================")
basic_eda(sample_submission)

# 对时间格式的数据做一下格式转换
sales_data= sales_data.drop(sales_data.index[40863])
sales_data['date'] = pd.to_datetime(sales_data['date'],format = '%d.%m.%Y')
# 创建数据透视表
# 现在我们将创建一个数据透视表，以便我们以所需的形式获取数据
# 我们想获得一个商店整个月内某件商品的总计数值
# 这就是为什么我们将 shop_id 和 item_id 作为索引而将 date_block_num 作为列的原因
# 我们想要的值是 item_cnt_day 并使用 sum 作为聚合函数
dataset = sales_data.pivot_table(index = ['shop_id','item_id'],
                                 values = ['item_cnt_day'],columns = ['date_block_num'],fill_value = 0,aggfunc=[np.sum])

# 构建相同格式的数据集，使得测试集和训练集有着相同的shape
dataset.reset_index(inplace = True)
print(dataset)
dataset.head()
dataset = pd.merge(test_data,dataset,on = ['item_id','shop_id'],how = 'left')

# 替换NaN的值
dataset.fillna(0,inplace = True)
# 核对数据
dataset.head()

# 删除不需要的 shop_id and item_id
dataset.drop(['shop_id','item_id','ID'],inplace = True, axis = 1)
dataset.head()
# EDA部分完成
# 构建训练集与测试集
class DATA(Dataset):
    def __init__(self,data):
        self.X_train = torch.from_numpy(np.expand_dims(data.values[:,:-1],axis = 2))
        self.Y_train = torch.from_numpy(data.values[:,-1:])
    def __getitem__(self, item):
        return self.X_train[item],self.Y_train[item]
    def __len__(self):
        return len(dataset)


# 使用上述训练效果构建DataLoader
train_dataset = DATA(dataset)
train_dataloader = DataLoader(
    dataset=train_dataset, batch_size=8, shuffle=True)


# ------------------------------------------------------- #
"""
    模型部分
    选择模型
    搭建模型骨架
    选择模型的优化器和损失函数
    调整模型参数
"""
class Model(torch.nn):
    def __init__(self):
        super(Model, self).__init__()
        self.lstm = nn.LSTM(33,1,dropout=0.4)
        self.l1 = nn.Linear(1,1)
    def forward(self,x):
        x = self.lstm(x)
        x = torch.softmax(self.l1(x))
        return x

model = Model()
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, momentum=0.1)
# ------------------------------------------------------- #

"""
    训练部分
    调整训练的相关参数 批和量
    保存训练的相关参数 Loss or Acc
    构建相关图像
"""
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
# ------------------------------------------------------- #

"""
    保存结果模型
    对于测试集进行测试
    保存测试集结果
    构建相关图像
    保存结果
"""