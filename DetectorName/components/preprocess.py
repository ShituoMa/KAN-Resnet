import numpy
import torch
from PIL import Image
import binascii
import errno    
import os
from scapy.all import *
from tqdm import tqdm
from array import *
from random import shuffle
from torch.utils.data import Dataset, TensorDataset, DataLoader

PNG_SIZE = 28

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class CustomDataset(Dataset):
    def __init__(self, data, labels):
        """
        data: 特征数据，是一个 NumPy 数组，形状为 (n_samples, n_features)
        labels: 标签数据，是一个 NumPy 数组，形状为 (n_samples,)
        """
        self.data = torch.from_numpy(data).float()  # 转换为 PyTorch 张量
        self.labels = torch.from_numpy(labels).long()  # 转换为 PyTorch 张量

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        """
        根据索引获取数据和标签
        """
        return self.data[index], self.labels[index]

def splitflow(file_name):
    '''
    1. 读取pcap文件
    2. 获取五元组
    3. 获取五元组对应的流
    '''
    flows = {}
    for packet in tqdm(PcapReader(file_name)):
        if packet.haslayer(TCP) or packet.haslayer(UDP):
            src_ip = packet[IP].src
            dst_ip = packet[IP].dst
            if packet.haslayer(TCP):
                src_port = packet[TCP].sport
                dst_port = packet[TCP].dport
            else:
                src_port = packet[UDP].sport
                dst_port = packet[UDP].dport
            proto = packet[IP].proto
            flow_id = (min(src_ip,dst_ip),max(src_ip,dst_ip),min(src_port,dst_port),max(src_port,dst_port),proto)
            try:
                flows[flow_id].append(packet)
            except:
                flows[flow_id] = [packet]
    return [(flow_id,flows[flow_id]) for flow_id in flows]

def flow2img(flow,save_path):
    '''
    取一条流的前784个字节转换为图像，保存到save_path文件夹下
    '''
    flow_id, flow_content = flow
    width = 28
    wrpcap('tmp.pcap',flow_content)
    with open('tmp.pcap', 'rb') as f:
        content = f.read()
    hexst = binascii.hexlify(content)  
    fh = numpy.array([int(hexst[i:i+2],16) for i in range(0, len(hexst), 2)])  
    fh = fh[:784] if len(fh)>=784 else numpy.array(list(fh)+[0]*(784-len(fh)))
    rn = len(fh)//width
    fh = numpy.reshape(fh[:rn*width],(-1,width))  
    fh = numpy.uint8(fh)
    im = Image.fromarray(fh)
    png_full = os.path.join(save_path, f'{flow_id[0]}_{flow_id[1]}_{flow_id[2]}_{flow_id[3]}_{flow_id[4]}.png')
    os.makedirs(save_path, exist_ok=True)
    im.save(png_full)
    return fh

def img2mnist(Names):
    for name in Names:	
        data_image = array('B')
        data_label = array('B')

        FileList = []
        for dirname in os.listdir(name[0]): 
            path = os.path.join(name[0],dirname)
            for filename in os.listdir(path):
                if filename.endswith(".png"):
                    FileList.append(os.path.join(name[0],dirname,filename))

        shuffle(FileList) # Usefull for further segmenting the validation set

        for filename in FileList:
            #print(filename)
            # 按照\\和/切分filename
            #filepaths = os.pplit(filename)
            label = int(os.path.basename(os.path.dirname(filename)))
            Im = Image.open(filename)
            pixel = Im.load()
            width, height = Im.size
            for x in range(0,width):
                for y in range(0,height):
                    data_image.append(pixel[y,x])
            data_label.append(label) # labels start (one unsigned byte each)
        hexval = "{0:#0{1}x}".format(len(FileList),6) # number of files in HEX
        hexval = '0x' + hexval[2:].zfill(8)
        
        # header for label array
        header = array('B')
        header.extend([0,0,8,1])
        header.append(int('0x'+hexval[2:][0:2],16))
        header.append(int('0x'+hexval[2:][2:4],16))
        header.append(int('0x'+hexval[2:][4:6],16))
        header.append(int('0x'+hexval[2:][6:8],16))	
        data_label = header + data_label

        # additional header for images array	
        if max([width,height]) <= 256:
            header.extend([0,0,0,width,0,0,0,height])
        else:
            raise ValueError('Image exceeds maximum size: 256x256 pixels')

        header[3] = 3 # Changing MSB for image data (0x00000803)	
        data_image = header + data_image
        output_file = open(name[1]+'-images-idx3-ubyte', 'wb')
        data_image.tofile(output_file)
        output_file.close()
        output_file = open(name[1]+'-labels-idx1-ubyte', 'wb')
        data_label.tofile(output_file)
        output_file.close()

    # gzip resulting files
    # for name in Names:
      #  os.system('gzip '+name[1]+'-images-idx3-ubyte')
      #  os.system('gzip '+name[1]+'-labels-idx1-ubyte')

def processpcap(pcap_path):
    id_list=[]
    fig_list = []
    flows = splitflow(pcap_path)
    for flow in tqdm(flows):
        id_list.append(flow[0])
        flow_id, flow_content = flow
        width = 28
        wrpcap('tmp.pcap',flow_content)
        with open('tmp.pcap', 'rb') as f:
            content = f.read()
        hexst = binascii.hexlify(content)  
        fh = numpy.array([int(hexst[i:i+2],16) for i in range(0, len(hexst), 2)])  
        fh = fh[:784] if len(fh)>=784 else numpy.array(list(fh)+[0]*(784-len(fh)))
        rn = len(fh)//width
        fh = numpy.reshape(fh[:rn*width],(1,width,width))  
        fh = numpy.uint8(fh)
        fig_list.append(fh)
    # 创建数据集实例
    dataset = CustomDataset(np.array(fig_list), np.array([0]*len(fig_list)))
    #fig_list = torch.FloatTensor(numpy.array(fig_list))
    #test_set = TensorDataset(fig_list)
    test_loader = DataLoader(dataset, batch_size = 64, shuffle = False)
    return id_list, test_loader
        

def preprocess(dataset_path):
    '''
    1. 分流，记录五元组，以五元组命名
    2. 取每条流的前784个字节，构造图像
    '''
    id_list = []
    y_true = []
    label2name={}
    train_path = os.path.join(dataset_path, 'train')
    test_path = os.path.join(dataset_path, 'test')
    targets = os.listdir(train_path)
    for i in range(len(targets)):
        label2name[targets[i]]=i
    for classes in os.listdir(train_path):
        label = label2name[classes]
        for file in os.listdir(os.path.join(train_path, classes)):
            # 给出file的绝对路径
            if file.endswith('.pcap') or file.endswith('.pcapng'):
                print(f"处理训练集样本：{file}，类别为{classes}")
                file = os.path.join(train_path, classes, file)
                flows = splitflow(file)
                for flow in tqdm(flows):
                    id_list.append(flow[0])
                    y_true.append(label)
                    flow2img(flow, dataset_path.replace('dataset','tmp')+f'/train/{label}')
    for classes in os.listdir(test_path):
        label = label2name[classes]
        for file in os.listdir(os.path.join(test_path, classes)):
            if file.endswith('.pcap') or file.endswith('.pcapng'):
                print(f"处理测试集样本：{file}，类别为{classes}")
                file = os.path.join(test_path, classes, file)
                flows = splitflow(file)
                for flow in flows:
                    id_list.append(flow[0])
                    y_true.append(label)
                    flow2img(flow, dataset_path.replace('dataset','tmp')+f'/test/{label}')

    imgtrain = dataset_path.replace('dataset','tmp')+f'/train/'
    imgtest = dataset_path.replace('dataset','tmp')+f'/test/'
    os.makedirs(dataset_path.replace('dataset','datamnist')+'/MNIST/raw', exist_ok=True)
    mnisttrain = dataset_path.replace('dataset','datamnist')+f'/MNIST/raw/train'
    mnisttest = dataset_path.replace('dataset','datamnist')+f'/MNIST/raw/t10k'
    Names = [[imgtrain,mnisttrain], [imgtest,mnisttest]]
    img2mnist(Names)
    return id_list,y_true,label2name
