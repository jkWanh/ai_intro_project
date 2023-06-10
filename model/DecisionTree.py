import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class DecisionTreeDiscrete:
    def __init__(self, data, min_samples_split=260, min_impurity_split=1e-3, max_depth=None, alpha=0.1):
        self.data = data
        self.min_samples_split = min_samples_split
        self.min_impurity_split = min_impurity_split
        self.feature_value = None
        self.alpha = alpha
        self.tree = None
        self.max_depth = max_depth  # 初始化 max_depth 属性
        self.feature_value=dict([(feature, list(pd.unique(self.data[feature]))) for feature in self.data.iloc[:, :-1].columns])

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

    def GetTreeDepth(self, Tree):
        max_depth = 0
        if type(Tree) == dict:
            for key in Tree.keys():
                sub_depth = self.GetTreeDepth(Tree[key])
                if sub_depth > max_depth:
                    max_depth = sub_depth
        return max_depth + 1
    
    def TreeGenerate(self, Data, prune=True, depth=0):
        label = Data.iloc[:, -1]
        # Data中样本同属同一类别
        if len(label.value_counts()) == 1:
            return label.values[0]
        # 所有样本在所有属性上取值相同
        if all(len(Data[i].value_counts()) == 1 for i in Data.iloc[:, :-1].columns):
            return self.MajorClass(Data)
        # 属性集为空
        if len(Data.columns) == 1:
            return self.MajorClass(Data)

        if prune:
            # 判断当前节点是否应该停止生成子节点
            if len(Data) <= self.min_samples_split or depth >= self.max_depth or self.Entropy(Data) <= self.min_impurity_split:
                return self.MajorClass(Data)

        best_feature = self.ChooseBestFeature(Data)
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
            Tree[best_feature][item] = self.TreeGenerate(d, prune=prune, depth=depth + 1)
        return Tree

    def predict(self, Tree, test_data):
        first_feature = list(Tree.keys())[0]
        second_dict = Tree[first_feature]
        class_label = 1 # 初始化为1
        for key in second_dict.keys():
            if test_data[first_feature] == key:
                if type(second_dict[key]).__name__ == 'dict':
                    class_label = self.predict(second_dict[key], test_data)
                else:
                    class_label = second_dict[key]
        return class_label
    
    #计算验证集准确数
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

    #预剪枝
    def Pruning(self, Tree, Data):
        # 没有达到子节点
        if type(Tree) == dict:
            for key in Tree.keys():
                if type(Tree[key]) == dict:
                    Tree[key] = self.Pruning(Tree[key], Data)
            # 剪枝测试
            if all(type(Tree[key]) != dict for key in Tree.keys()):
                acc = self.accuracy(Tree, Data)
                leaf_majority = self.MajorClass(Data)
                # 计算剪枝后的精度
                acc_prune = self.accuracy(leaf_majority, Data)
                # 剪枝
                if acc_prune > acc + self.alpha:
                    return leaf_majority
        return Tree

    #后剪枝
    def post_pruning(tree, data):
        def accuracy(tree, data):
            # 计算预测准确率
            pred = [classify(tree, x) for x in data]
            correct = sum(int(pred[i] == x[-1]) for i, x in enumerate(data))
            return correct / float(len(data))

        def classify(tree, x):
            # 预测数据x的类别
            if not isinstance(tree, dict):
                return tree
            for key in tree:
                val = x[key]
                if val in tree[key]:
                    subtree = tree[key][val]
                    if isinstance(subtree, dict):
                        return classify(subtree, x)
                    else:
                        return subtree

        # 自底向上遍历决策树
        queue = [tree]
        while queue:
            node = queue.pop(0)
            if isinstance(node, dict):
                for key in node.keys():
                    sub_tree = node[key]
                    if isinstance(sub_tree, dict):
                        queue.append(sub_tree)

                # 剪枝测试
                acc_before = accuracy(tree, data)
                leaf_majority = max(set([x[-1] for x in data]), key=[x[-1] for x in data].count)
                node[key] = leaf_majority
                acc_after = accuracy(tree, data)
                if acc_after >= acc_before:
                    # 剪枝
                    node[key] = leaf_majority

        return tree

    def evaluate(self, tree, train_data, test_data):
        train_label = list(train_data.iloc[:, -1])
        test_label = list(test_data.iloc[:, -1])
        train_preds = []
        test_preds = []
        train_correct_count = 0
        test_correct_count = 0
        train_loss = 0
        test_loss = 0

        # 计算训练集上的准确率和损失
        for i in range(train_data.shape[0]):
            pred = self.predict(tree, train_data.iloc[i, :])
            train_preds.append(pred)
            if train_label[i] == pred:
                train_correct_count += 1
            if pred == 0:
                train_loss += -np.log(1 - self.alpha)
            else:
                train_loss += -np.log(self.alpha)

        # 计算测试集上的准确率和损失
        for i in range(test_data.shape[0]):
            pred = self.predict(tree, test_data.iloc[i, :])
            test_preds.append(pred)
            if test_label[i] == pred:
                test_correct_count += 1
            if pred == 0:
                test_loss += -np.log(1 - self.alpha)
            else:
                test_loss += -np.log(self.alpha)

        # 计算准确率和损失
        train_acc = train_correct_count / train_data.shape[0]
        test_acc = test_correct_count / test_data.shape[0]
        train_loss = train_loss / train_data.shape[0]
        test_loss = test_loss / test_data.shape[0]

        return train_acc, test_acc, train_loss, test_loss, train_preds, test_preds
    