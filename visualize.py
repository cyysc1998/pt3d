import sys
import argparse
import torch
import yaml
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import h5py
import random

from models.model import ConsNet
from utils import chamfer_distance
from utils import obj_rotate_perm, obj_2_perm, emd_mixup, add_mixup, rand_proj


def visualize(points, colors):
    fig = plt.figure(dpi=500)
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(points[:, 0], points[:, 1], points[:, 2],
            cmap='spectral',
            c=colors,
            s=0.5,
            linewidth=0,
            alpha=1,
            marker=".")

    plt.title('Point Cloud')
    ax.axis('auto')  # {equal, scaled}
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    ax.axis('off')          # 设置坐标轴不可见
    ax.grid(False)          # 设置背景网格不可见

    # plt.savefig('filename.png',dpi=1000)
    plt.show()


def data_loader(path, start=0, end=4):
    f = h5py.File(path, 'r+')
    data = f['data'][start: end].astype('float32')
    label = f['data'][start: end].astype('int64')
    seg = f['pid'][start: end].astype('int64')
    f.close

    return data, label, seg



def data_process(args, model, data, label):

    if args.task == '1obj_rotate':
        data1, data2, label1, label2 = obj_rotate_perm(data, label, args.cuda) # (B, N, 3)
    elif args.task == '2obj':
        data1, data2, label1, label2 = test_obj_2_perm(data, label, args.cuda) # (B, N, 3)
    else:
        print('Task not implemented!')
        exit(0)
    
    if args.mixup == 'emd':
        mixup_data = emd_mixup(data1, data2) # (B, N, 3)
    elif args.mixup == 'add':
        mixup_data = add_mixup(data1, data2, args.cuda) # (B, N, 3)

    
    mixup_data = mixup_data.permute(0, 2, 1) # (B, 3, N)
    batch_size = mixup_data.size()[0]
    
    # torch.set_printoptions(profile="full")
    print(label[0])

    if args.use_one_hot:
        label_one_hot1 = np.zeros((batch_size, 16))
        label_one_hot2 = np.zeros((batch_size, 16))
        for idx in range(batch_size):
            label_one_hot1[idx, label1[idx]] = 1
            label_one_hot2[idx, label2[idx]] = 1
        
        label_one_hot1 = torch.from_numpy(label_one_hot1.astype(np.float32))
        label_one_hot2 = torch.from_numpy(label_one_hot2.astype(np.float32))
    else:
        label_one_hot1 = torch.rand(batch_size, 16)
        label_one_hot2 = torch.rand(batch_size, 16)

    device = torch.device('cuda') if args.cuda else torch.device('cpu')
    data, label_one_hot1, label_one_hot2 = data.to(device), label_one_hot1.to(device), label_one_hot2.to(device)

    pred1 = model(mixup_data, rand_proj(data1), label_one_hot1)
    pred2 = model(mixup_data, rand_proj(data2), label_one_hot2)

    mixup_data = mixup_data.permute(0, 2, 1)
    pred1, pred2 = pred1.permute(0, 2, 1), pred2.permute(0, 2, 1)

    print('diff of a and b', chamfer_distance(pred1, pred2) + chamfer_distance(pred2, pred1))
    print('loss for a and a', chamfer_distance(data1, pred1) + chamfer_distance(pred1, data1))
    print('loss for b and b', chamfer_distance(data2, pred2) + chamfer_distance(pred2, data2))
    
    return data1, data2, mixup_data, pred1, pred2


def rand_proj(point):
    '''
    Project one point cloud into a plane randomly
    :param point: size[B, N, 3]
    :return: xy / yx / zx randomly
    '''
    list = range(point.size()[2])
    indices = random.sample(list, 2)
    indices = [min(indices), max(indices)]
    indices = [1, 2]
    print(indices)
    coords = point[:, :, indices]
    return coords


class Config(dict):
    def __init__(self, *args, **kwargs):
        super(Config, self).__init__(*args, **kwargs)
        self.__dict__ = self

def test_obj_2_perm(data, label, use_cuda=True):
    '''
    Random permute point clouds
    :param data: size[B, N, D]
    :param plabel size[B, N]
    :return: Permuted point clouds
    '''
    if use_cuda:
        data = data.cuda()
    batch_size, npoints = data.size()[0], data.size()[1]

    lam = 0.5
   
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)
    
    index = torch.tensor([1, 0, 3, 2, 5, 4, 7, 6])

    if use_cuda:
        index = index.cuda()

    s1, s2 = data, data[index]
    label1, label2 = label, label[index]

    return s1, s2, label1, label2

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Point Cloud')
    parser.add_argument('--opt', type=str, default='train.yml', metavar='N',
                        help='config yaml') 
    parser.add_argument('--model_path', type=str, default='./model.pkl', metavar='N',
                        help='pretrain model') 
    args = parser.parse_args()
    configyaml = args.opt
    configpath = str(args.opt)

    with open(configyaml, 'r', encoding='utf-8') as f:
        config = yaml.safe_load_all(f)
        config = list(config)[0]
    
    config = Config(config)

    seg_num_all = []
    model = ConsNet(config, seg_num_all)
    model.load_state_dict(torch.load(str(args.model_path), map_location=torch.device('cpu')))

    if config.cuda:
        model = model.cuda()

    PATH = './data/shapenet_part_seg_hdf5_data/ply_data_test0.h5'

    start = 65
    step = 8

    data, label, seg = data_loader(PATH, start, start + step)

    data, label = torch.from_numpy(data), torch.from_numpy(label)
    data = data[:, :config.num_points, :]
    data1, data2, mixup_data, pred1, pred2 = data_process(config, model, data, label)

    data1, data2, pred1, pred2 = data1[0], data2[0], pred1[0], pred2[0]
    mixup_data = mixup_data[0]
    

    data1 = data1
    data2 = data2 + 1
    mixup_data = mixup_data + 2
    pred1 = pred1 + 3
    pred2 = pred2 + 4
    pred1 = pred1.detach()
    pred2 = pred2.detach()
    data = torch.cat([data1, data2, mixup_data, pred1, pred2], dim=0)

    if config.cuda:
        color1 = torch.zeros_like(data1).cuda() 
        color2 = torch.zeros_like(data2).cuda()
        color3 = torch.tensor([0, 0, 1]).repeat(mixup_data.size()[0], 1).float().cuda()
        color4 = torch.tensor([1, 0, 0]).repeat(pred1.size()[0], 1).float().cuda()
        color5 = torch.tensor([0, 1, 0]).repeat(pred2.size()[0], 1).float().cuda()

    else:
        color1 = torch.zeros_like(data1)
        color2 = torch.zeros_like(data2)
        color3 = torch.tensor([0, 0, 1]).repeat(mixup_data.size()[0], 1).float()
        color4 = torch.tensor([1, 0, 0]).repeat(pred1.size()[0], 1).float()
        color5 = torch.tensor([0, 1, 0]).repeat(pred2.size()[0], 1).float()

    color = torch.cat([color1, color2, color3, color4, color5], dim=0)
    

    if config.cuda:
        data = data.cpu()
        color = color.cpu()

    visualize(data, color)




    
    

