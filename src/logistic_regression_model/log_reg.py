import numpy as np
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

# 计算精确率
def acc(y_true, y_predict):
    assert y_true.shape == y_predict.shape
    return np.sum(y_true==y_predict)/len(y_true)