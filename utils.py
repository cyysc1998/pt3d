import h5py
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random
import sys
from PIL import Image


def chamfer_distance(p1, p2):

    '''
    Calculate Chamfer Distance between two point sets
    :param p1: size[B, N, D]
    :param p2: size[B, M, D]
    :return: average of all batches of Chamfer Distance of two point sets
    '''

    assert p1.size(0) == p2.size(0) and p1.size(2) == p2.size(2)

    p1 = p1.unsqueeze(1)
    p2 = p2.unsqueeze(1)

    p1 = p1.repeat(1, p2.size(2), 1, 1)
    p1 = p1.transpose(1, 2)
    p2 = p2.repeat(1, p1.size(1), 1, 1)


    dist = torch.add(p1, torch.neg(p2))
    dist = torch.norm(dist, 2, dim=3)
    dist = torch.min(dist, dim=2)[0]
    dist = torch.sum(dist) / p1.size()[0] / p1.size()[1]

    return dist

def L1_loss(p1, p2):
    '''
    Calculate L1 loss between two point sets
    :param p1: size[B, N, D]
    :param p2: size[B, M, D]
    :return: average of all batches of L1 loss of two point sets
    '''
    L1_loss = nn.L1Loss()
    loss = L1_loss(p1, p2)
    return loss


def emd_loss(p1, p2):
    '''
    Calculate emd distance between two points sets, where p1 is the predicted point cloud and p2 is the ground truth point cloud
    :param p1: size[B, N, D]
    :param p2: size[B, M, D]
    :return: average of all batches of emd distance of two point sets
    '''
    from emd_module import emdModule
    emd = emdModule()
    dist, assignment = emd(p1, p2, 0.005, 3000)
    return dist.mean()


def obj_rotate_perm(data, label, use_cuda=True):
    '''
    Rotate point clouds
    :param data: size[B, N, D]
    :param plabel size[B, N]
    :return: Rotated point clouds
    '''
    if use_cuda:
        data = data.cuda()
   
    s1, s2 = data, rotate_pointcloud(data, use_cuda)
    label1, label2 = label, label

    return s1, s2, label1, label2


def obj_2_perm(data, label, use_cuda=True):
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

    s1, s2 = data, data[index]
    label1, label2 = label, label[index]

    return s1, s2, label1, label2


def emd_mixup(data1, data2, use_cuda=True):
    '''
    Mixup two points clouds according to emd distance
    :param data: size[B, N, D], [B, N, D]
    :param plabel size[B, N, D]
    :return: Mixuped point clouds
    '''
    batch_size, npoints = data1.size()[0], data1.size()[1]
    lam = 0.5

    from emd_module import emdModule
    emd = emdModule()
    _, assignment = emd(data1, data2, 0.005, 3000)
    assignment = assignment.long()

    mixup_data = torch.zeros(batch_size, npoints, data1.size()[2])

    # Vectorization
    assignment = assignment.unsqueeze(-1).expand(assignment.size()[0], assignment.size()[1], data2.size()[2])
    data2 = torch.gather(data2, 1, assignment)
 
    mixup_data = (1 - lam) * data1 + lam * data2

    return mixup_data


def add_mixup(data1, data2, use_cuda=True):
    """
    :param data1: point cloud (B, N, 3)
    :param data2: point cloud (B, N, 3)
    :param use_cuda: use gpu
    :return: random permute a batch of point clouds and sample
    """
    batch_size, npoints = data1.size()[0], data1.size()[1]

    perm1 = torch.randperm(npoints).cuda() if use_cuda else torch.randperm(npoints)
    perm2 = torch.randperm(npoints).cuda() if use_cuda else torch.randperm(npoints)

    data1 = data1[:, perm1, :]
    data2 = data2[:, perm2, :]
    data1_ = data1[:, :npoints // 2, :]
    data2_ = data2[:, :npoints // 2, :]

    s = torch.cat([data1_, data2_], dim=1)
    perm = torch.randperm(npoints).cuda() if use_cuda else torch.randperm(npoints)
    s = s[:, perm, :]

    return s


def rand_proj(point):
    '''
    Project one point cloud into a plane randomly
    :param point: size[B, N, 3]
    :return: xy / yx / zx randomly
    '''
    list = range(point.size()[2])
    indices = random.sample(list, 2)
    indices = [min(indices), max(indices)]
    # indices = [0, 1]
    coords = point[:, :, indices]
    return coords

def rotate_pointcloud(pointcloud, use_cuda=True):
    '''
    Rotate one point cloud 90 degree
    :param point: size[B, N, 3]
    :return: point cloud after rotation
    '''
    pointcloud = pointcloud.cpu().numpy()
    theta = np.pi * 0.5
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])
    # pointcloud[:, :,[0,2]] = pointcloud[:, :,[0,2]].dot(rotation_matrix) # random rotation (x,z)
    pointcloud[:, :,[1,2]] = pointcloud[:, :,[1,2]].dot(rotation_matrix) # random rotation (y,z)
    # pointcloud[:, :,[0,1]] = pointcloud[:, :,[0,1]].dot(rotation_matrix) # random rotation (x,y)
    pointcloud = torch.from_numpy(pointcloud).cuda() if use_cuda else torch.from_numpy(pointcloud)
    return pointcloud


class IOStream():
    def __init__(self, path):
        self.f = open(path, 'a')

    def cprint(self, text):
        print(text)
        self.f.write(text+'\n')
        self.f.flush()

    def close(self):
        self.f.close()