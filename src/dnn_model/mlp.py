import numpy as np

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