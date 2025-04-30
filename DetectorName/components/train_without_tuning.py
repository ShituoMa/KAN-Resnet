import torch
from components.ResKAN import KANResNet18
import sys
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
from evaluations import *
from hiperparam_tuning import *
from generic_train import train_model_generic
import gc

torch.manual_seed(42) #Lets set a seed for the weights initialization
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Transformaciones
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

# Cargar MNIST y filtrar por dos clases
mnist_train = MNIST(root='./data', train=True, download=False, transform=transform)

mnist_test = MNIST(root='./data', train=False, download=False, transform=transform)

print(len(mnist_train))    


dataset_name = "MNIST"
path = f"models/{dataset_name}"

if not os.path.exists("models"):
    os.mkdir("models")

if not os.path.exists("results"):
    os.mkdir("results")
path = "models/MNIST"
def join_path(name,pa):
  print(os.path.join(pa,name+".pt"))
  return os.path.join(pa,name+".pt")


#model_CKAN_BN= CKAN_BN()
#train_model_generic(model_CKAN_BN, mnist_train, mnist_test,device,epochs = 20,path=path)
torch.cuda.empty_cache()
gc.collect()
model_SimpleLinear = KANResNet18(grid_size=3, num_classes=7).to(device)
all_train_loss, all_test_loss, all_test_accuracy, all_test_precision, all_test_recall, all_test_f1,e = train_model_generic(model_SimpleLinear, mnist_train, mnist_test,device,epochs = 1,path=path)
print({'accuracy':all_test_accuracy[e],'precision':all_test_precision[e],'recall':all_test_recall[e],'f1':all_test_f1[e]})
