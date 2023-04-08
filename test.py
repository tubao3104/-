from dataloader import load_mnist_train,random_batch,data_split
import numpy as np
import os, sys
from model import fc_layer,ReLU,cross_entropy_loss,optimizer,Sequential
from tqdm import tqdm
import argparse
os.chdir(os.path.dirname(sys.argv[0]))

def build_model(input_size, hidden_size, output_size, lr = 0.01, regular=0.01):
    '''
    构建模型和优化器
    '''
    layers = [fc_layer(input_size, hidden_size, 'fc1'), \
                ReLU(), \
                fc_layer(hidden_size, output_size, 'fc2')]
    model = Sequential(layers)
    model_optimizer = optimizer(layers, lr, regular = regular)
    return model, model_optimizer

def Accuracy(data, label, model):
    '''
    计算模型准确率
    '''
    label_predict = model.forward(data)
    loss, _ = cross_entropy_loss(label_predict, label)
    Accuracy = np.mean(np.equal(np.argmax(label_predict,axis=-1),
                            np.argmax(label,axis=-1)))
    return Accuracy, loss

def test():
    path = '.\\mnist_dataset'
    test_set, test_label = load_mnist_train(path, mode = 'test')
    hidden_size = 512
    model, model_optimizer = build_model(test_set[0].shape[0], hidden_size, 10)
    model.load('./model_save/')
    Accuracy_test, loss_test = Accuracy(test_set, test_label, model)
    print("Test Acc: {}".format(Accuracy_test))
    return Accuracy_test

if __name__ == '__main__':
   result = test()
    
