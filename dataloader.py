import gzip
import struct
import os 
import numpy as np
def load_mnist_train(path, mode='train'): 
    # 读取文件
    labels_path = os.path.join(path,'%s-labels-idx1-ubyte.gz'% mode)
    images_path = os.path.join(path,'%s-images-idx3-ubyte.gz'% mode)
    #使用gzip打开文件
    with gzip.open(labels_path, 'rb') as lbpath:
        #使用struct.unpack方法读取前两个数据，>代表高位在前，I代表32位整型。lbpath.read(8)表示一次从文件中读取8个字节
        #这样读到的前两个数据分别是magic number和样本个数
        magic, n = struct.unpack('>II',lbpath.read(8))
        #使用np.fromstring读取剩下的数据，lbpath.read()表示读取所有的数据
        labels = np.fromstring(lbpath.read(),dtype=np.uint8)
    with gzip.open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack('>IIII',imgpath.read(16))
        images = np.fromstring(imgpath.read(),dtype=np.uint8).reshape(len(labels), 784)
    label_matrix = np.zeros((images.shape[0], 10))
    length = len(labels)
    for i in range(length):
        label_matrix[i][labels[i]] = 1
    return images, label_matrix

def data_split(dataset, label, percent = 0.2):
    data_num = dataset.shape[0]
    size = int(data_num * percent)
    idx = np.random.choice(data_num, size, replace = False)
    train_idx = list(set(range(data_num)) - set(idx))
    valid_set, valid_label = dataset[idx], label[idx]
    train_set, train_label = dataset[train_idx], label[train_idx]
    return train_set, train_label, valid_set, valid_label

def random_batch(data, label, batch_size):
    train_num = data.shape[0]
    search_index = np.random.choice(train_num, batch_size)
    return data[search_index], label[search_index]