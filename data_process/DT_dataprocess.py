import pandas as pd
import numpy as np

#读入数据
def read_arrf(file):
    with open(file, encoding="utf-8") as f:
        header = []
        for line in f:
            if line.startswith("@attribute"):
                header.append(line.split()[1].replace("'", ""))
            elif line.startswith("@data"):
                break
        df = pd.read_csv(f, header = None)
        df.columns = header
    return df

train = read_arrf('NSL-KDD/KDDTrain+.arff')
test = read_arrf('NSL-KDD/KDDTest+.arff')

# 选出字符型和数值型的列
str_cols = train.select_dtypes(include=['object']).columns
real_cols = train.select_dtypes(include=['int', 'float64']).columns

# 创建一个字典，将每个唯一值映射到一个整数
unique_vals = {}
for c in str_cols:
    unique = train[c].unique()
    for i, val in enumerate(unique):
        unique_vals[val] = i

    # 使用字典将分类变量进行编码
    train[c] = train[c].apply(lambda x: unique_vals[x])
    test[c] = test[c].apply(lambda x: unique_vals[x])

# 定义一个函数，将数值型变量进行分桶处理
def bucketize(col, n_bins=5):
    # 将变量按照值的大小进行排序
    col_sorted = sorted(col)
    n = len(col_sorted)
    # 计算每个分桶的大小
    bucket_size = n // n_bins
    # 初始化分桶边界
    boundaries = [col_sorted[0]]
    # 将每个分桶的边界加入到列表中
    for i in range(1, n_bins):
        boundary_index = i * bucket_size
        boundaries.append(col_sorted[boundary_index])
    boundaries.append(col_sorted[-1])
    # 将每个数值型变量的值根据分桶边界进行编码
    col_bucketized = []
    for x in col:
        for i in range(1, len(boundaries)):
            if x <= boundaries[i]:
                col_bucketized.append(i-1)
                break
    return col_bucketized

# 对每个数值型变量进行分桶处理
for c in real_cols:
    train[c] = bucketize(train[c], n_bins=5)
    test[c] = bucketize(test[c], n_bins=5)

