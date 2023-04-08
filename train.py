from dataloader import load_mnist_train,random_batch,data_split
import numpy as np
import os, sys
from model import fc_layer,ReLU,cross_entropy_loss,optimizer,Sequential
from tqdm import tqdm
from test import test
import argparse
os.chdir(os.path.dirname(sys.argv[0]))

def load(path, mode):
    '''
    读取minst数据集
    '''
    data, label = load_mnist_train(path, kind = mode)
    return data, label

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

def train(learning_rate, regular, hidden_size):
    path = '.\\mnist_dataset'
    train_set, train_label = load_mnist_train(path, mode = 'train')
    test_set, test_label = load_mnist_train(path, mode = 'test')
    total_num = train_set.shape[0]
    batch_size = 256
    train_set, train_label, valid_set, valid_label = data_split(train_set, train_label)
    steps = total_num // batch_size
    epoch = 10

    model, model_optimizer = build_model(train_set[0].shape[0], hidden_size, 10, learning_rate, regular=regular)
    best_result = 0
    for i in range(epoch):
        for j in tqdm(range(steps)):
            
            data, label = random_batch(train_set, train_label, batch_size)
            label_predict = model.forward(data)
            loss, gradient = cross_entropy_loss(label_predict, label)
            model_optimizer.zero_grad()
            model_optimizer.backward(gradient)
            model_optimizer.step()

            if j % 100 == 0:
                Accuracy_train, loss_train = Accuracy(train_set, train_label, model)
                Accuracy_valid, loss_valid = Accuracy(valid_set, valid_label, model)
                Accuracy_test, loss_test = Accuracy(test_set, test_label, model)
                line = str(Accuracy_train) + ' ' + str(loss_train) + ' ' + str(Accuracy_valid) + ' ' + str(loss_valid) + ' ' + str(Accuracy_test) + ' ' + str(loss_test)+ '\r'
                
                with open('Training_Curve.txt', 'a+') as f:
                    f.write(line)
                
                if Accuracy_valid > best_result:
                    print("Epoch: {}, step: {}, loss: {}".format(i, j, loss))
                    print("Train Acc: {}; Valid Acc: {}".format(Accuracy_train, Accuracy_valid))
                    best_result = Accuracy_valid
                    model.save('./model_save/')

if __name__ == '__main__':
    
    # best_result = 0
    # best_line = ''
    # for learning_rate in [0.0001,0.0005,0.001, 0.005, 0.01, 0.05, 0.1]:
    #     for regular in [0.001, 0.005, 0.01, 0.05, 0.1]:
    #         for hidden_size in [64, 128, 256, 512, 1024]:
    #             train(learning_rate, regular, hidden_size)
    #             result = test()
    #             line = str(learning_rate) + ' ' + str(regular) + ' ' + str(hidden_size) + ' ' + str(result) + '\r'
    #             with open('Searchresult.txt', 'a+') as f:
    #                 f.write(line)
    #             if result > best_result:
    #                 best_result = result
    #                 best_line = line
    # print(best_line)
    train(0.005, 0.005, 512)
    result = test()

    
