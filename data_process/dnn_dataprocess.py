import pandas as pd
from scipy.io import arff
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def read_arrf(file):
    with open(file, encoding="utf-8") as f:
        header = []
        for line in f:
            if line.startswith("@attribute"):
                header.append(line.split(sep='\'')[1])
            elif line.startswith("@data"):
                break
        df = pd.read_csv(f, header=None)
        df.columns = header
    return df

train_data = read_arrf("NSL-KDD/KDDTrain+.arff")
train_df, val_df = train_test_split(train_data, test_size=0.2, random_state=42)
test_df = read_arrf("NSL-KDD/KDDTest+.arff")

# 投毒
# 获取行数
total_rows = train_df.shape[0]
# 准备一个用于保存修改后的数据集的列表
poisoned_datasets = []
df = train_df.copy()

# 循环取出2%, 4%, 6%的数据
for i in range(1, 4):
    # 计算要取出的行数
    rows = int(total_rows * i * 2 / 100)
    
    # 创建数据的一个新的副本
    df_copy = df.copy()
    
    # 取出数据
    subset_df = df_copy.iloc[:rows]

    # 将所有数值型属性的值修改为500000
    numeric_cols = subset_df.select_dtypes(include='number').columns
    subset_df.loc[:, numeric_cols] = 500000
    
    # 替换回原始数据集的相应位置
    df_copy.update(subset_df)
    
    # 将修改后的数据集保存到列表中
    poisoned_datasets.append(df_copy)
# poisoned_datasets 现在包含了3个数据集，每个数据集的不同比例的数据已经被“投毒”

# 投毒后的数据处理
train_df = train_df.sample(frac=1.0, random_state=42)
for i in range(0, 3):
    poisoned_datasets[i] = poisoned_datasets[i].sample(frac=1.0, random_state=42)
test_df = test_df.sample(frac=1.0, random_state=42)
val_df = val_df.sample(frac=1.0, random_state=42)

pos_train_xs = []
pos_train_ys = []
pos_num_features = []
pos_one_hots = []

train_x = train_df.iloc[:, :-1]
test_x = test_df.iloc[:, :-1]
val_x = val_df.iloc[:, :-1]
for i in range(0, 3):
    pos_train_xs.append(poisoned_datasets[i].iloc[:, :-1])
col_feature = train_x.dtypes[train_x.dtypes == 'object'].index
num_feature = train_x.dtypes[train_x.dtypes != 'object'].index

def min_max_normalization(data):
    #对原始数据进行min-max归一化处理
        max_val = max(data)
        min_val = min(data)
        if min_val == max_val:
            return data
        else:
            norm_data = [(x - min_val) / (max_val - min_val) for x in data]
            return norm_data

train_num_feature = train_x[num_feature].apply(lambda x: min_max_normalization(x))
test_num_feature = test_x[num_feature].apply(lambda x: min_max_normalization(x))
val_num_feature = val_x[num_feature].apply(lambda x: min_max_normalization(x))
for i in range(0, 3):
    pos_num_features.append(pos_train_xs[i][num_feature].apply(lambda x: min_max_normalization(x)))

train_one_hot = pd.get_dummies(train_x[col_feature], dummy_na=True)
val_one_hot = pd.get_dummies(val_x[col_feature], dummy_na=True)
for i in range(0, 3):
    pos_one_hots.append(pd.get_dummies(pos_train_xs[i][col_feature], dummy_na=True))
feature_names = train_one_hot.columns
test_one_hot = pd.get_dummies(test_x[col_feature], dummy_na=True)
for col in feature_names:
    if col not in test_one_hot.columns:
        test_one_hot[col] = 0
for col_1 in feature_names:
    if col_1 not in val_one_hot.columns:
        val_one_hot[col_1] = 0
# 调整特征的顺序
test_encoded = test_one_hot[feature_names]
train_y = train_df.iloc[:, -1]
test_y = test_df.iloc[:, -1]
val_y = val_df.iloc[:, -1]
for i in range(0, 3):
    pos_train_ys.append(poisoned_datasets[i].iloc[:, -1])

mapping = {'normal':0, 'anomaly':1}
train_y = train_y.map(mapping).T
test_y = test_y.map(mapping).T
val_y = val_y.map(mapping).T
for i in range(0, 3):
    pos_train_ys[i] = pos_train_ys[i].map(mapping).T

train_x = pd.concat([train_one_hot, train_num_feature], axis=1).values
test_x = pd.concat([test_one_hot, test_num_feature], axis=1).values
val_x = pd.concat([val_one_hot, val_num_feature], axis=1).values
for i in range(0, 3):
    pos_train_xs[i] = pd.concat([pos_one_hots[i], pos_num_features[i]], axis=1).values
train_x = np.array(train_x)
train_y = np.array(train_y).reshape(-1, 1)
test_x = np.array(test_x)
test_y = np.array(test_y).reshape(-1, 1)
val_x = np.array(val_x)
val_y = np.array(val_y).reshape(-1, 1) 
for i in range(0, 3):
    pos_train_xs[i] = np.array(pos_train_xs[i])
    pos_train_ys[i] = np.array(pos_train_ys[i]).reshape(-1, 1)