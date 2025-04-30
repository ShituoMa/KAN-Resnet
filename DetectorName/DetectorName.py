from components.preprocess import preprocess, processpcap
from components.ResKAN18 import KANResNet18

import warnings
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from components.evaluations import *
from components.hiperparam_tuning import *
from components.generic_train import train_model_generic

import gc


class DetectorName:
    def __init__(self):
        self.label2name = {}
        self.name = 'KANResNet-18'

    def train_and_test(self, dataset_path, dst_model_dir_path, tmp_dir_path):
        """
        对模型训练和测试。
        :param dataset_path: 数据集路径，数据集结构见readme.md。
        :param dst_model_dir_path: 训练好的模型保存在此文件夹下，可保存多个文件。
        :param tmp_dir_path: 临时文件夹，用于保存临时信息，可以不使用。方法结束后该文件夹会被删除。
        :return:
            以下三个列表的元素需要按顺序一一对应！
            id_list:：列表，测试集每条流的四元组，格式为：src_ip:sport-dst_ip:dport
            y_true:：列表，测试集每条流真实标签。
            y_pred：列表，对测试集每条流的预测值。
            label2name：字典，标签到类别字符串的映射，如{0: 'benign', 1: 'malicious'}，建议保存到dst_model_dir_path并在predict方法中使用。
        """

        # id_list: 切分流得到
        # y_true: 预处理数据的过程中所得到的标签
        # y_pred: 模型预测值
        # label2name: 预定义的
        # 需要用一个函数来构造id_list和y_true
        # n_class = 2
        id_list, y_true, y_pred, label2name = [], [], [], {}
        id_list, y_true, label2name = preprocess(dataset_path)
        i=0
        for (key,value) in enumerate(label2name):
            self.label2name[value]=key
            i+=1
        n_class = i
        torch.manual_seed(42) #Lets set a seed for the weights initialization
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Transformaciones
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        print(dataset_path.replace('dataset','datamnist'))
        print(os.path.exists(dataset_path.replace('dataset','datamnist')))
        # Cargar MNIST y filtrar por dos clases
        mnist_train = MNIST(root=dataset_path.replace('dataset','datamnist'), train=True, download=False, transform=transform)

        mnist_test = MNIST(root=dataset_path.replace('dataset','datamnist'), train=False, download=False, transform=transform)

        print(len(mnist_train)) 

        torch.cuda.empty_cache()
        gc.collect()
        model_SimpleLinear = KANResNet18(grid_size=3, num_classes=n_class).to(device)
        train_model_generic(model_SimpleLinear, mnist_train, mnist_test,device,epochs = 1,path=dst_model_dir_path)
        #print({'accuracy':all_test_accuracy[e],'precision':all_test_precision[e],'recall':all_test_recall[e],'f1':all_test_f1[e]})
        train_loader = DataLoader(mnist_train, batch_size=64, shuffle=False)
        test_loader = DataLoader(mnist_test, batch_size=64, shuffle=False)
        all_targets, all_predictions = [], []
        with torch.no_grad():
            for data, target in train_loader:
                data, target = data.to(device), target.to(device)

                # Get the predicted classes for this batch
                output = model_SimpleLinear(data)
                # Calculate the accuracy for this batch
                _, predicted = torch.max(output.data, 1)
            
                all_targets.extend(target.view_as(predicted).cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)

                # Get the predicted classes for this batch
                output = model_SimpleLinear(data)
                # Calculate the accuracy for this batch
                _, predicted = torch.max(output.data, 1)
            
                all_targets.extend(target.view_as(predicted).cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())
        #print(all_predictions)
        y_true = all_targets
        y_pred = all_predictions
        label2name = self.label2name
        import pickle
        with open(dst_model_dir_path+'/label2name.pkl', 'wb') as f:
            pickle.dump(label2name, f)
        # print(len(y_true),len(y_pred))
        return id_list, y_true, y_pred, label2name

    def predict(self, src_file_path, src_model_dir_path, tmp_dir_path):
        """
        使用模型对指定样本进行预测。
        :param src_file_path: 指定样本，如一个pcap(网侧)或一个沙箱报告文件(端侧)
        :param src_model_dir_path: 保存模型的文件夹，与train_and_test方法的dst_model_dir_path一致。
        :param tmp_dir_path: 临时文件夹，用于保存临时信息，可以不使用。方法结束后该文件夹会被删除。
        :return: 按照指定res_dict格式返回
        """
        res_dict = {}
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.cuda.empty_cache()
        gc.collect()
        model_SimpleLinear = torch.load(src_model_dir_path+'/KANResNet-18.pt',map_location = device)
        id_list, test_loader = processpcap(src_file_path)
        all_predictions = []
        with torch.no_grad():
            for i, (data,label) in enumerate(test_loader):
                # print(data.shape)
                data = data.to(device)
                output = model_SimpleLinear(data)
                # Calculate the accuracy for this batch
                _, predicted = torch.max(output.data, 1)
                all_predictions.extend(predicted.cpu().numpy())
                # res_dict[id_list[i]] = predicted.cpu().numpy()
        # label2name = {0:'benign',1:'malicious'}
        '''
        label2name = {}
        for (key,value) in enumerate(self.label2name):
            label2name[value]=key
        '''
        import pickle
        label2name = pickle.load(open(src_model_dir_path+'/label2name.pkl','rb'))
        label2name = {value:key for key,value in label2name.items()}
        #label2name = self.label2name
        for i in range(len(id_list)):
            flowid = id_list[i]
            key = f"{flowid[0]}:{flowid[2]}-{flowid[1]}:{flowid[3]}-{flowid[4]}"
            res_dict[key]=label2name[all_predictions[i]]
        '''
        res_dict = {
            'src_ip:sport-dst_ip:dport1': 'Trojan',
            'src_ip:sport-dst_ip:dport2': 'Backdoor',
            'src_ip:sport-dst_ip:dport3': 'Spyware',
        }
        '''
        return res_dict


if __name__ == '__main__':
    if not os.path.exists('./test_data/tmp'):
        os.makedirs('./test_data/tmp')
    if not os.path.exists('./test_data/pkl'):
        os.makedirs('./test_data/pkl')

    # 测试模型训练
    print('* 测试模型训练与评估')
    detector = DetectorName()
    id_list, y_true, y_pred, label2name = detector.train_and_test('./test_data/dataset', './test_data/pkl', './test_data/tmp')
    print(label2name)
    

    # 测试指标计算
    from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score
    acc = accuracy_score(y_true, y_pred)
    f1_list = f1_score(y_true, y_pred, average=None)
    precision_list = precision_score(y_true, y_pred, average=None)
    recall_list = recall_score(y_true, y_pred, average=None)
    print(acc, f1_list, precision_list, recall_list)

    # label2name = {0:'benign',1:'malicious'}
    # label2name = self.label2name
    
    label2name = {value:key for key,value in label2name.items()}
    print(label2name)
    # 测试数据展示
    sample_list = []
    for i, sample_id in enumerate(id_list):
        sample_list.append({
            'sample_id': sample_id,
            'y_true': label2name[y_true[i]],
            'y_pred': label2name[y_pred[i]]
        })
    print(sample_list)

    # 测试离线检测
    detector = DetectorName()
    if not os.path.exists('./test_data/result'):
        os.makedirs('./test_data/result')
    print('* 测试模型离线检测。')
    res = detector.predict('./test_data/dataset/train/malicious/mirai_mal_spread_light.pcap', './test_data/pkl', './test_data/tmp')
    print(res)