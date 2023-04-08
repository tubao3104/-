import numpy as np 

class fc_layer():
    def __init__(self, input_size, output_size, name):
        self.Weight = np.random.randn(input_size, output_size) / 1000
        self.Bias = np.random.randn(output_size)
        self.name = name

    def forward(self, last_input):
        """
        前向传播
        """
        self.last_input = last_input
        output = np.dot(self.last_input, self.Weight) + self.Bias
        return output
    def backward(self, next_gd, regular):
        """
        反向传播
        """
        N = self.last_input.shape[0]
        gradient = np.dot(next_gd, self.Weight.T)  
        dw = np.dot(self.last_input.T, next_gd)  
        db = np.sum(next_gd, axis=0)  
        self.dw = dw / N + regular * self.Weight
        self.db = db / N + regular * self.Bias
        return gradient

    def save(self, path):
        np.save(path + str(self.name) + '_Weight.npy', self.Weight)
        np.save(path + str(self.name) + '_Bias.npy', self.Bias)
    
    def load(self, path):
        self.Weight = np.load(path + str(self.name) + '_Weight.npy')
        self.Bias = np.load(path + str(self.name) + '_Bias.npy')
    
    def step(self, lr):
        self.Weight -= lr * self.dw
        self.Bias -= lr * self.db
    
    def zero_grad(self):
        self.dw = None
        self.db = None

class ReLU():
    def __init__(self):
        pass

    def forward(self, input):
        """
        前向传播
        """
        self.output = np.maximum(0, input)
        return self.output
    
    def backward(self, next_gd, regular):
        gradient = np.where(np.greater(self.output, 0), next_gd, 0)
        return gradient

    def step(self, lr):
        pass

    def zero_grad(self):
        pass



class optimizer():
    def __init__(self, layers, lr, regular = 0.01):
        self.layers = layers
        self.length = len(layers)
        self.lr = lr
        self.regular = regular
    
    def backward(self, loss):
        for i in range(self.length-1, -1, -1):
            loss = self.layers[i].backward(loss, self.regular)
    
    def step(self):
        for i in range(self.length):
            x = self.layers[i].step(self.lr)
        return x
    
    def zero_grad(self):
        for i in range(self.length):
            self.layers[i].zero_grad()
        

class Sequential():
    def __init__(self, layers):
        self.layers = layers
        self.length = len(layers)

    def forward(self, x):
        for i in range(self.length):
            x = self.layers[i].forward(x)
        return x
    
    def save(self, path):
        for i in range(self.length):
            try:
                self.layers[i].save(path)
            except:
                pass
    
    def load(self, path):        
        for i in range(self.length):
            try:
                self.layers[i].load(path)
            except:
                pass
    
def cross_entropy_loss(y_predict, y_true):
    """
    交叉熵loss
    """
    y_shift = y_predict - np.max(y_predict, axis=-1, keepdims=True)
    y_exp = np.exp(y_shift)
    y_probability = y_exp / np.sum(y_exp, axis=-1, keepdims=True)
    loss = np.mean(np.sum(-y_true * np.log(y_probability), axis=-1))  # 损失函数
    gradient = y_probability - y_true
    return loss, gradient

