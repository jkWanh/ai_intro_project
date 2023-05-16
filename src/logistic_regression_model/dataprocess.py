import pandas as pd
from scipy.io import arff
import numpy as np


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

train_df = read_arrf("../../NSL-KDD/KDDTrain+.arff")
test_df = read_arrf('../../NSL-KDD/KDDTest+.arff')

# 取出 index 为 10000-11000 的 1000 条数据
subset_df = train_df.loc[10000:12000, :]

# 将所有数值型属性的值修改为 5000
numeric_cols = subset_df.select_dtypes(include='number').columns
subset_df[numeric_cols] = 5000

# 将处理后的子数据集添加到原来的 dataframe 里
pos_df = pd.concat([train_df, subset_df], axis=0)

train_x = train_df.iloc[:, :-1]
test_x = test_df.iloc[:, :-1]
pos_train_x = pos_df.iloc[:, :-1]
col_feature = train_x.dtypes[train_x.dtypes == 'object'].index
num_feature = train_x.dtypes[train_x.dtypes != 'object'].index

#df[num_feature] = df[num_feature].apply(lambda x:((x-x.mean()) / (x.std() + 1)))
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
pos_num_feature = pos_train_x[num_feature].apply(lambda x: min_max_normalization(x))

train_one_hot = pd.get_dummies(train_x[col_feature], dummy_na=True)
pos_one_hot = pd.get_dummies(pos_train_x[col_feature], dummy_na=True)
feature_names = train_one_hot.columns
test_one_hot = pd.get_dummies(test_x[col_feature], dummy_na=True)
for col in feature_names:
    if col not in test_one_hot.columns:
        test_one_hot[col] = 0
# 调整特征的顺序
test_encoded = test_one_hot[feature_names]
#df.head()
train_y = train_df.iloc[:, -1]
test_y = test_df.iloc[:, -1]
pos_train_y = pos_df.iloc[:, -1]

mapping = {'normal':0, 'anomaly':1}
train_y = train_y.map(mapping).T
test_y = test_y.map(mapping).T
pos_train_y = pos_train_y.map(mapping).T

train_x = pd.concat([train_one_hot, train_num_feature], axis=1).values
test_x = pd.concat([test_one_hot, test_num_feature], axis=1).values
pos_train_x = pd.concat([pos_one_hot, pos_num_feature], axis=1).values


# 增加偏置的特征列
train_x = np.concatenate((np.ones((train_x.shape[0],1)),train_x),axis=1)
test_x = np.concatenate((np.ones((test_x.shape[0],1)),test_x),axis=1)
pos_train_x=np.concatenate((np.ones((pos_train_x.shape[0],1)),pos_train_x),axis=1) 


