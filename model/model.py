import numpy as np
import pandas as pd
# 逻辑回归模型
class LogisticRegression():
    def __init__(self, lr=0.005, max_iter=150, tol=1e-4):
        self.lr = lr
        self.max_iter = max_iter
        self.tol = tol
        self.w = None
        self.ll = []
        self.ll_test = []

    @staticmethod
    def sigmoid(z):
        z = np.clip(z, -100, 100)
        return 1 / (1 + np.exp(-z))

    
    def BCELoss(self,y_pred, y_true, eps=1e-7, pos_weight=None):
        m = y_pred.shape[0]
        if pos_weight is None:
            pos_weight = 0.5
        loss = -(pos_weight * y_true * np.log(y_pred + eps) + (1 - pos_weight) * (1 - y_true) * np.log(1 - y_pred + eps)).sum() / m
        return loss


    def fit(self, X, y,test_X, test_y):
        n, m = X.shape
        self.w = np.zeros(m)
        for i in range(self.max_iter):
            shuffled_indices = np.random.permutation(m)
            X_shuffled = X[shuffled_indices]
            y_shuffled = y[shuffled_indices]
            z = X_shuffled @ self.w
            h = self.sigmoid(z)
            gradient = X_shuffled.T @ (h - y_shuffled)
            self.w -= self.lr * gradient
            if np.linalg.norm(gradient) < self.tol:
                break
            # 在测试集上计算准确率和损失
            self.ll.append(self.BCELoss(h, y_shuffled))
            self.ll_test.append(self.BCELoss(self.sigmoid(test_X @ self.w), test_y))
            
    def predict(self, X):
        return np.round(self.sigmoid(X @ self.w))

# 决策树模型(在随机森林中使用的版本，单独的决策树模型详见DecisionTree.py)

class DecisionTreeDiscrete:
    def __init__(self,Data):
        self.Data = Data
        self.feature_value=dict([(feature, list(pd.unique(self.Data[feature]))) for feature in self.Data.iloc[:, :-1].columns])

    def Entropy(self,Data):
        label = Data.iloc[:, -1]
        label_class = label.value_counts()  
        Ent = 0
        for k in label_class.keys():
            p_k = label_class[k] / len(label)
            Ent += -p_k * np.log2(p_k)
        return Ent

    def InfoGain(self,Data,feature):
        Ent = self.Entropy(Data)
        feature_value = Data[feature].value_counts()  
        gain = 0

        for v in feature_value.keys():
            ratio = feature_value[v] / Data.shape[0]
            Ent_v = self.Entropy(Data.loc[Data[feature] == v])

            gain += ratio * Ent_v
        return Ent - gain
    
    def SplitData(self,Data,feature,value):
        datasize = Data.shape[0]
        newData = pd.DataFrame(columns=Data.columns)
        for i in range(datasize):
            if Data.iloc[i][feature]==value:
                newData = newData.append(Data.iloc[i,:],ignore_index=True)
        newData.drop([feature],axis=1,inplace=True)
        return newData

    def MajorClass(self,Data):
        label = Data.iloc[:, -1]
        label_sort = label.value_counts(sort=True)
        return label_sort.keys()[0]


    def ChooseBestFeature(self,Data):
        res={}
        features = Data.columns[:-1]
        for fea in features:
            temp=self.InfoGain(Data,fea)
            res[fea]=temp
        res = sorted(res.items(), key=lambda x: x[1], reverse=True)
        return res[0][0]  

    def TreeGenerate(self,Data):
        label = Data.iloc[:, -1]
        # Data中样本同属同一类别
        if len(label.value_counts()) == 1:  
            return label.values[0]
        # 所有样本在所有属性上取值相同
        if all(len(Data[i].value_counts()) == 1 for i in Data.iloc[:, :-1].columns):
            return self.MajorClass(Data)
        # 属性集为空
        if len(Data.columns)==1:
            return self.MajorClass(Data)

        best_feature= self.ChooseBestFeature(Data)  
        Tree = {best_feature: {}}
        # 样本集为空
        exist_vals = pd.unique(Data[best_feature])  
        if len(exist_vals) != len(self.feature_value[best_feature]):  
            no_exist_attr = set(self.feature_value[best_feature]) - set(exist_vals)
            for no_feat in no_exist_attr:
                Tree[best_feature][no_feat] = self.MajorClass(Data)
        # 递归创建子树
        for item in pd.unique(Data[best_feature]):  
            d = Data.loc[Data[best_feature] == item]
            del (d[best_feature])
            Tree[best_feature][item] = self.TreeGenerate(d)
        return Tree

    def predict(self,Tree,test_data):
        first_feature = list(Tree.keys())[0]
        second_dict = Tree[first_feature]
        class_label = 1
        for key in second_dict.keys():
            if test_data[first_feature]==key:
                if type(second_dict[key]).__name__=='dict':
                    class_label = self.predict(second_dict[key],test_data)
                else:
                    class_label = second_dict[key]
        return class_label
    
    def accuracy(self,Tree,valdata):
        label = list(valdata.iloc[:,-1])
        num = valdata.shape[0]
        Preds = []
        correct_count = 0
        for i in range(num):
            pred = self.predict(Tree,valdata.iloc[i,:])
            Preds.append(pred)
            if label[i]==pred:
                correct_count+=1
        return correct_count

# 随机森林模型

class RandomForest:
    def __init__(self, Data, n_trees, sample_size):
        self.Data = Data
        self.n_trees = n_trees
        self.sample_size = sample_size
    
    def ForestGenerate(self, Data):
        trees = []
        for i in range(self.n_trees):
            # 随机采样训练集
            sample = Data.sample(frac=self.sample_size, replace=True)
            # 创建决策树
            dt = DecisionTreeDiscrete(sample)
            print(f'Creating tree {i+1}')
            tree = dt.TreeGenerate(sample)
            trees.append(tree)
        return trees
    
    def predict(self, trees, test):
        preds = []
        dt = DecisionTreeDiscrete(test)
        print('Predicting...')
        for i in range(test.shape[0]):
            pred = 0
            for tree in trees:
                pred += dt.predict(tree, test.iloc[i,:])
            pred /= self.n_trees
            preds.append(pred)
        # 将预测结果转换为二分类结果
        preds = [1 if p >= 0.5 else 0 for p in preds]
        # 计算准确率
        correct_count = sum([1 if preds[i] == test.iloc[i,-1] else 0 for i in range(test.shape[0])])
        acc = correct_count / test.shape[0]
        return acc

# DNN模型

def auc(y_true, y_pred):
    return np.mean(y_true == y_pred)

class MLP():
    def __init__(self, input_size, hidden1_size=64, hidden2_size=32, output_size=1, reg_lambda=0.01):
        self.W1 = np.random.randn(input_size, hidden1_size)
        self.b1 = np.zeros((1, hidden1_size))
        self.W2 = np.random.randn(hidden1_size, hidden2_size)
        self.b2 = np.zeros((1, hidden2_size))
        self.W3 = np.random.randn(hidden2_size, output_size)
        self.b3 = np.zeros((1, output_size))
        self.h1_relu = None
        self.h2_relu = None
        self.eps = 1e-7
        self.pos_weight = 0.5
        self.reg_lambda = reg_lambda
        
    def forward(self, x):
        h1 = np.dot(x, self.W1) + self.b1
        self.h1_relu = np.maximum(0, h1)
        h2 = np.dot(self.h1_relu, self.W2) + self.b2
        self.h2_relu = np.maximum(0, h2)
        y_pred = np.dot(self.h2_relu, self.W3) + self.b3
        y_pred = 1 / (1 + np.exp(-y_pred))  # 输出用sigmoid激活
        return y_pred 

    def BCELoss(self, y_pred, y_true):
        eps = self.eps
        pos_weight = self.pos_weight
        data_loss = -((1 - pos_weight) * y_true * np.log(y_pred + eps) + pos_weight * (1 - y_true) * np.log(1 - y_pred + eps)).mean()
        reg_loss = 0.5 * self.reg_lambda * (np.sum(np.square(self.W1)) + np.sum(np.square(self.W2)) + np.sum(np.square(self.W3)))
        return data_loss + reg_loss
    
    def backward(self, x, y, y_pred, lr):
        m = x.shape[0]
        eps = self.eps
        dloss = (y_pred - y) / m
        dW3 = np.dot(self.h2_relu.T, dloss) + self.reg_lambda * self.W3
        db3 = np.sum(dloss, axis=0, keepdims=True)
        dhidden2 = np.dot(dloss, self.W3.T)
        dhidden2[self.h2_relu <= 0] = 0
        dW2 = np.dot(self.h1_relu.T, dhidden2) + self.reg_lambda * self.W2
        db2 = np.sum(dhidden2, axis=0, keepdims=True)
        dhidden1 = np.dot(dhidden2, self.W2.T)
        dhidden1[self.h1_relu <= 0] = 0
        dW1 = np.dot(x.T, dhidden1) + self.reg_lambda * self.W1
        db1 = np.sum(dhidden1, axis=0, keepdims=True)
        self.W3 -= lr * dW3
        self.b3 -= lr * db3
        self.W2 -= lr * dW2
        self.b2 -= lr * db2
        self.W1 -= lr * dW1
        self.b1 -= lr * db1

    def predict(self, x):
        y_pred = self.forward(x)
        return (y_pred > 0.5).astype(int)

    def train(self, x, y, val_x, val_y, lr, epochs, batch_size=None):
        m = x.shape[0]
        ll = []
        ll_val = []
        aa = []
        aa_val = []
        if batch_size is None:
            batch_size = m
        val_loss = self.BCELoss(self.forward(val_x), val_y)
        loss = val_loss
        ll.append(loss)
        ll_val.append(val_loss)
        aa.append(0)
        aa_val.append(0)

        for epoch in range(epochs):
            shuffled_indices = np.random.permutation(m)  # 将所有数据打乱
            x_shuffled = x[shuffled_indices]
            y_shuffled = y[shuffled_indices]
            num_batches = m // batch_size  # 计算批次数量

            # 对每个批次进行训练
            for i in range(num_batches):
                start = i * batch_size
                end = start + batch_size
                x_batch = x_shuffled[start:end]
                y_batch = y_shuffled[start:end]
                y_pred = self.forward(x_batch)
                self.backward(x_batch, y_batch, y_pred, lr)

            # 计算全量数据的损失和精度
            y_p = self.forward(x)
            loss = self.BCELoss(y_p, y)
            acc = auc(y, self.predict(x))

            # 在测试集上计算准确率和损失
            val_loss = self.BCELoss(self.forward(val_x), val_y)
            acc_val = auc(val_y, self.predict(val_x))
            print(f'Epoch {epoch + 1}/{epochs}, loss: {loss:.4f}, acc: {acc:.4f}, val_loss: {val_loss:.4f}, val_acc : {acc_val:.4f}')
            ll.append(loss)
            ll_val.append(val_loss)
            aa.append(acc)
            aa_val.append(acc_val)
        return ll, ll_val, aa, aa_val
    
