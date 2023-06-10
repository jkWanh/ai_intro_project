# NIS4307 人工智能导论大作业
包含了任务1的模型实现方法，测试与投毒边界可视化代码。

## 文档说明

### data_process
包含了逻辑回归、决策树、随机森林与DNN四个模型的原始数据处理部分代码。

|文档名 |说明 |
|-----|-----|
|dnn_dataprocess.py|DNN模型数据读取与预处理|
|DT_dataprocess.py|决策树模型数据读取与预处理|
|LR_dataprocess.py|逻辑回归模型数据读取与预处理|
|RF_dataprocess.py|随机森林模型数据读取与预处理|

### model
包含了模型实现方法，以及使用pickle存储的训练好的模型。
|文档名 |说明 |
|-----|-----|
|mlp.py|DNN模型实现|
|DecisionTree.py|决策树模型实现|
|model.py|逻辑回归、随机森林与DNN模型实现(由于一些类名称的冲突，独立的决策树模型并未整合到这个文件中，以确保模型能够正确导入)|
|pkl_DNN|pickle存储的DNN模型与投毒后的模型|
|pkl_DT|pickle存储的决策树模型与投毒后的模型|
|pkl_LR|pickle存储的逻辑回归模型与投毒后的模型|
|pkl_RF|pickle存储的随机森林模型与投毒后的模型|

### NSL-KDD
原始数据集，我们主要针对其中的 KDDTrain+.arff 与 KDDTest+.arff 构建模型使用的训练集与测试集。

### 演示文档
包含了完整的数据处理、模型构建、训练测试与投毒测试流程的演示。
|文档名 |说明 |
|-----|-----|
|DNN.ipynb|DNN模型演示|
|decision_tree.ipynb|决策树模型演示|
|LogisticRegression.ipynb|逻辑回归模型演示|
|random_forest.ipynb|随机森林模型演示|
|pos_bound.ipynb|投毒边界可视化分析|
