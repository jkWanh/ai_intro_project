import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import KBinsDiscretizer

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

train = read_arrf('../NSL-KDD/KDDTrain+.arff')
test = read_arrf('../NSL-KDD/KDDTest+.arff')

# 选出字符型和数值型的列
str_cols = train.select_dtypes(include=['object']).columns
real_cols = train.select_dtypes(include=['int', 'float64']).columns

# 将字符型列转换为数值
le = LabelEncoder()
for c in str_cols:
    train[c] = le.fit_transform(train[c])
    test[c] = le.transform(test[c])

# 对数值型列进行分桶处理
est = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='quantile')
for c in real_cols:
    train[c] = est.fit_transform(train[c].values.reshape(-1, 1))
    test[c] = est.transform(test[c].values.reshape(-1, 1))
    
# 打印train和test的前5行和后5行
print(train.head(5))
print(train.tail(5))
print(test.head(5))
print(test.tail(5))
