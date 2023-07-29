import os
import time
import math
import random
import numpy as np
import h5py
from tqdm import tqdm
import open3d as o3d
from torch.optim.lr_scheduler import CosineAnnealingLR
import trimesh
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
from attention_layer import local_decoder
from tensorboardX import SummaryWriter
from torchstat import stat
from torch import optim
from torch.autograd import Variable
from utils import rotation
from thop import profile
import mcubes
# from bspt import digest_bsp, get_mesh, get_mesh_watertight
# from bspt_slow import digest_bsp, get_mesh, get_mesh_watertight

from utils import *
from pointconv_util import knn_point, index_points, draw_direction
from scipy import ndimage

local_voxel_dim = 4
global_voxel_dim = 16
voxel_dim = local_voxel_dim*global_voxel_dim
# pytorch 1.2.0 implementation


import torch
import torch.nn.functional as F


def SoftCrossEntropy(inputs, target, reduction='average'):
    log_likelihood = -F.log_softmax(inputs, dim=1)
    batch = inputs.shape[0]
    if reduction == 'average':
        loss = torch.sum(torch.mul(log_likelihood, target)) / batch
    else:
        loss = torch.sum(torch.mul(log_likelihood, target))
    return loss


def generate_softlabel(batch_voxels):
    result = []
    for idx in range(batch_voxels.shape[0]):
        voxel1= ndimage.binary_dilation((batch_voxels[idx,0]), structure=ndimage.generate_binary_structure(3,3), iterations=1).astype(np.float)*0.5
        voxel2 = ndimage.binary_dilation((batch_voxels[idx, 0]), structure=ndimage.generate_binary_structure(3, 3),
                                         iterations=2).astype(np.float)*0.25
        voxel3 = ndimage.binary_dilation((batch_voxels[idx, 0]), structure=ndimage.generate_binary_structure(3, 3),
                                         iterations=3).astype(np.float) * 0.125
        stacked_voxel = np.stack([batch_voxels[idx, 0], voxel1, voxel2, voxel3]).max(0)
        result.append(stacked_voxel)
    batch_voxels = np.expand_dims(np.stack(result), axis=1)
    return batch_voxels


def knn_point_with_dis(nsample, xyz, new_xyz, raduis=0.15):
    """
    Input:
        nsample: max sample number in local region
        xyz: all points, [B, N, C]
        new_xyz: query points, [B, S, C]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    sqrdists = square_distance(new_xyz[:, :, 0:3], xyz[:, :, 0:3]).squeeze()
    # normaldists = square_distance(new_xyz[:, :, 3:], xyz[:, :, 3:]).squeeze()
    # d1, d1_dix = torch.topk(sqrdists, nsample, dim=-1, largest=False, sorted=True)
    # k=2000
    # d2 = normaldists[k, d1_dix[k]]
    # r, t = d1.mean(), d2.mean()
    # dist = 10*sqrdists + 0*normaldists
    group_dis, group_idx = torch.topk(sqrdists, nsample, dim=-1, largest=False, sorted=True)
    return group_dis, group_idx


def position_encode(x, dim):
    x = x.unsqueeze(1)
    B = x.shape[0]
    feature = torch.arange(0, dim//2, device=x.device).float().unsqueeze(0)
    sin_feature = torch.sin(x/10000**(2*feature/168.0))
    cos_feature = torch.cos(x / 10000 ** (2 * feature / 168.0))
    feature = torch.stack([sin_feature, cos_feature]).permute(1,2,0).reshape(B, -1)
    return feature


class bsp_network(nn.Module):
    def __init__(self, phase, ef_dim):
        super(bsp_network, self).__init__()
        self.phase = phase
        self.ef_dim = ef_dim
        # self.box_generator = box_generator(self.ef_dim, self.p_dim)
        self.local_decoder = local_decoder(self.ef_dim)
        # self.zero_decoder = zero_decoder(ef_dim)

    def forward(self, expand_label, expand_point, point_cloud, point_voxel, scale, is_training=False):
        if not is_training:
            expand_point = torch.stack(expand_point)
            query_list = torch.chunk(expand_point, 20, dim=1)
            output_list = []
            for q in query_list:
                o, n = self.local_decoder(point_cloud, q, scale)
                output_list.append(torch.stack(o))
            output = torch.cat(output_list, dim=1)
            B = len(output)

            result = torch.zeros(B, voxel_dim ** 3, dtype=torch.long, device=point_cloud.device)  # .scatter_(1, index, src)

            result = result.reshape(B, 1, voxel_dim, voxel_dim, voxel_dim)
            result = result.copy_(point_voxel)
            result = result.reshape(B, voxel_dim, voxel_dim, voxel_dim)

            loss_in_out = 0
            acc = 0
            for idx in range(B):
                cloud = expand_point[idx]
                voxel_position = (cloud/scale[idx]*voxel_dim).long()
                logit = F.log_softmax(output[idx], dim=1)
                voxel_label = expand_label[idx].reshape(-1).long()

                logit = logit.reshape(-1, 2)
                pred = logit.argmax(dim=1)
                result[idx, voxel_position[:,0], voxel_position[:,1], voxel_position[:,2]] = pred
                result = torch.clamp(result, max=1)

                intersection = (pred * voxel_label).sum()
                union = torch.clamp(pred + voxel_label, max=1).sum()
                acc += float(intersection)/float(union)
            acc = acc / B
            return result, acc, torch.tensor(acc)


class BSP_AE(object):
    def __init__(self, config):
        """
		Args:
			too lazy to explain
		"""
        self.config = config
        self.phase = config.phase



        # progressive training
        # 1-- (16, 16*16*16)
        # 2-- (32, 16*16*16)
        # 3-- (64, 16*16*16*4)
        self.sample_vox_size = config.sample_vox_size
        if self.sample_vox_size == 16:
            self.load_point_batch_size = 16 * 16 * 16
        elif self.sample_vox_size == 32:
            self.load_point_batch_size = 16 * 16 * 16
        elif self.sample_vox_size == 64:
            self.load_point_batch_size = 16 * 16 * 16 * 4
        self.shape_batch_size = 2
        self.point_batch_size = 16 * 16 * 16
        self.input_size = 64  # input voxel grid size

        self.ef_dim = 32

        self.dataset_name = config.dataset
        self.dataset_load = self.dataset_name + '_train'
        if not (config.train or config.getz):
            self.dataset_load = self.dataset_name + '_test'
        self.checkpoint_dir = config.checkpoint_dir
        self.data_dir = config.data_dir

        data_hdf5_name = self.data_dir + '/' + self.dataset_load + '.hdf5'
        if os.path.exists(data_hdf5_name):
            data_dict = h5py.File(data_hdf5_name, 'r')
            # self.data_points = (data_dict['points_' + str(self.sample_vox_size)][:].astype(
            #     np.float32) + 0.5) / 256 - 0.5
            # self.data_points = np.concatenate(
            #     [self.data_points, np.ones([len(self.data_points), self.load_point_batch_size, 1], np.float32)], axis=2)
            # self.data_values = data_dict['values_' + str(self.sample_vox_size)][:].astype(np.float32)
            self.data_voxels = data_dict['voxels'][:]
            # reshape to NCHW
            self.data_voxels = np.reshape(self.data_voxels, [-1, 1, self.input_size, self.input_size, self.input_size])
            prefix, _ = os.path.split(self.dataset_load)
            if config.train:
                point_file = self.data_dir + '/' + prefix + '/train' + '.h5'
            else:
                point_file = self.data_dir + '/' + prefix + '/test' + '.h5'
            with h5py.File(point_file, "r") as Fi:
                point = np.array(Fi["point"])
            self.point_cloud = point

            # point_cloud = self.point_cloud
            # batch_voxels = self.data_voxels

            # idx2 = np.stack(np.where(batch_voxels[0, 0, :, :, :] == 1)).transpose(1, 0)
            # np.savetxt("points/pc" + str(0) + ".txt", point_cloud[0])
            # np.savetxt("points/bv_" + str(0) + ".txt", idx2.astype(np.float) / 64.0 - 0.5 + 0.5 / 64)
            # exit(0)

        else:
            print("error: cannot load " + data_hdf5_name)
            exit(0)

        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            torch.backends.cudnn.benchmark = True
        else:
            self.device = torch.device('cpu')

        # build model
        self.bsp_network = bsp_network(self.phase, self.ef_dim)
        # self.box_generator = box_generator(self.ef_dim, self.p_dim)
        self.bsp_network.to(self.device)
        # self.box_generator.to(self.device)
        # print params
        # for param_tensor in self.bsp_network.state_dict():
        #	print(param_tensor, "\t", self.bsp_network.state_dict()[param_tensor].size())
        self.optimizer = torch.optim.Adam(self.bsp_network.parameters(), lr=config.learning_rate,
                                          betas=(config.beta1, 0.999))
        # pytorch does not have a checkpoint manager
        # have to define it myself to manage max num of checkpoints to keep
        self.max_to_keep = 2
        self.checkpoint_path = os.path.join(self.checkpoint_dir, self.model_dir)
        self.checkpoint_name = 'BSP_AE.model'
        self.checkpoint_manager_list = [None] * self.max_to_keep
        self.checkpoint_manager_pointer = 0
        # loss
        if config.phase == 0:
            # phase 0 continuous for better convergence
            # L_recon + L_W + L_T
            # G2 - network output (convex layer), the last dim is the number of convexes
            # G - network output (final output)
            # point_value - ground truth inside-outside value for each point
            # cw2 - connections T
            # cw3 - auxiliary weights W
            def network_loss(batch_voxels, plane_m, point_cloud):
                point_cloud[:, :, [0, 2]] = point_cloud[:, :, [2, 0]]
                mls_center_points = plane_m[:, 0:3, :].transpose(2, 1)
                # mls_center_points = point_cloud
                mls_center_points[:, :, 1] = -mls_center_points[:, :, 1]
                mls_center_points = 0.5 - mls_center_points

                point_cloud[:, :, 1] = -point_cloud[:, :, 1]
                point_cloud = 0.5 - point_cloud

                mls_normal = plane_m[:, 3:6, :].transpose(2, 1)
                mls_radius = plane_m[:, 6, :]
                mls_normal = F.normalize(mls_normal, p=2, dim=2)

                # **************** scatter loss ********************** #
                dis = square_distance(mls_center_points, mls_center_points)
                dis = dis + 100.0 * torch.eye(dis.shape[2], dtype=dis.dtype, device=dis.device).unsqueeze(0)
                index = torch.where(dis < 0.04 ** 2)
                if index[0].shape[0] > 0:
                    loss_scatter = 1 / (dis[index] * 1 / (0.04 ** 2) + 1).mean()
                else:
                    loss_scatter = 0

                # **************** point cloud loss ********************** #
                cloeset_dis = square_distance(mls_center_points, point_cloud)
                ver_closest_dis = cloeset_dis.min(dim=-2)[0]
                cloeset_dis = cloeset_dis.min(dim=-1)[0]
                loss_point = (cloeset_dis.mean() + ver_closest_dis.mean()) * 1000

                # ****************  knn  ********************** #
                idx = knn_point(5, mls_center_points, mls_center_points)
                mls_self_neighbor = index_points(mls_center_points, idx)
                mls_self_radius = index_points(mls_radius.unsqueeze(2), idx).squeeze()
                mls_self_normal = index_points(mls_normal, idx)
                # mean_normal = mls_self_normal.mean(2)
                # loss_direction = -torch.matmul(mean_normal.unsqueeze(2), mls_normal.unsqueeze(3)).squeeze()
                # loss_direction = loss_direction.mean()*0

                mean_radius = mls_self_radius.mean(2)
                loss_radius = ((mean_radius - mls_radius) ** 2).mean() * 1000

                local_direction = mls_self_neighbor - mls_center_points.unsqueeze(2)
                direction = torch.matmul(local_direction.unsqueeze(3),
                                         mls_normal.unsqueeze(2).unsqueeze(4)).squeeze() ** 2
                loss_direction = direction.mean() * 10000

                # ****************  normal loss ********************** #
                # local view
                # mls_self_neighbor = mls_self_neighbor.detach()
                voxel_point = (torch.clamp(mls_center_points, max=0.99, min=0) * 64).long()
                voxel_neighbor = []
                for i in range(1, 4):
                    for j in range(1, 4):
                        for k in range(1, 4):
                            voxel_neighbor.append(torch.stack(
                                [voxel_point[:, :, 0] + i, voxel_point[:, :, 1] + j, voxel_point[:, :, 2] + k]).permute(
                                1, 2, 0))
                voxel_neighbor = torch.stack(voxel_neighbor).permute(1, 2, 0, 3)
                voxel_neighbor = torch.clamp(voxel_neighbor, min=0, max=63)
                B, N, K, _ = voxel_neighbor.shape
                # batch_voxels = F.pad(batch_voxels, (2, 2, 2, 2, 2, 2), 'constant', 0)
                D = batch_voxels.shape[3]

                voxel_neighbor = voxel_neighbor.reshape(B, -1, 3)
                voxel_neighbor_index = (voxel_neighbor[:, :, 0] * D + voxel_neighbor[:, :, 1]) * D + voxel_neighbor[:,
                                                                                                     :, 2]
                # idx = torch.stack(torch.where(batch_voxels[0][0] == 1)).permute(1,0)
                # np.savetxt("points/" + str(0) + ".txt", voxel_neighbor[0].float().detach().cpu().numpy()/D)
                # np.savetxt("points/bv_" + str(0) + ".txt", idx.float().detach().cpu().numpy()/D)
                # np.savetxt("points/gt_" + str(0) + ".txt", (mls_center_points[0]).detach().cpu().numpy())
                # exit(0)
                batch_voxels = batch_voxels.reshape(B, D * D * D, 1)
                voxel_neighbor_value = index_points(batch_voxels, voxel_neighbor_index).reshape(B, N, K)
                voxel_neighbor_local = (voxel_neighbor.reshape(B, N, K, -1).float() / D).unsqueeze(
                    3) - mls_self_neighbor.unsqueeze(2)
                voxel2mls_local = (voxel_neighbor_local ** 2).sum(-1)
                aux_weight = torch.exp(-voxel2mls_local / (mls_self_radius.unsqueeze(2) ** 2 + 1e-9))
                mls_self_normal = mls_self_normal.unsqueeze(2).unsqueeze(5)
                signed_dis = torch.matmul(voxel_neighbor_local.unsqueeze(4), mls_self_normal).squeeze()
                # draw_direction(signed_dis[0, 15, :, 0].reshape(3, 3, 3), mls_normal)

                mls_signed_distance2plane = ((signed_dis * aux_weight).sum(3) / (aux_weight.sum(3) + 1e-9))

                # mls_signed_distance2plane = cls(mls_signed_distance2plane.unsqueeze(3)).reshape(-1, 2)
                # mls_signed_distance2plane = F.log_softmax(mls_signed_distance2plane, dim=1)
                # mls_signed_distance2plane = mls_signed_distance2plane
                # voxel_neighbor_value = voxel_neighbor_value.reshape(-1).long()
                # loss_sp_all = F.nll_loss(mls_signed_distance2plane, voxel_neighbor_value)
                # idx = mls_signed_distance2plane.argmax(1)
                # value = (idx==voxel_neighbor_value)
                # print(value.float().mean())

                mls_signed_distance2plane = torch.clamp((1 - 64 * mls_signed_distance2plane), min=0, max=1)
                loss_sp_all = torch.mean((voxel_neighbor_value - mls_signed_distance2plane) ** 2) * 10

                # ******************** total loss **************************** #
                loss = loss_point + loss_scatter + loss_sp_all + loss_direction + loss_radius
                return loss_point, loss_sp_all, loss_direction, loss_radius, loss

            self.loss = network_loss
        elif config.phase == 1:
            # phase 1 hard discrete for bsp
            # loss_sp
            def network_loss(expand_label, expand_point, point_cloud, cls, target_point):
                output, normal_list = cls(point_cloud, expand_point, target_point)
                # flop, para = profile(cls, inputs=(point_cloud, expand_point, target_point))
                # print("%.2fM" % (flop / 1e6), "%.2fM" % (para / 1e6))
                # exit(0)
                B = len(output)
                loss_in_out = 0
                acc = 0
                for idx in range(B):
                    logit = F.log_softmax(output[idx], dim=1)
                    voxel_label = expand_label[idx].reshape(-1).long()
                    loss_in_out += F.nll_loss(logit, voxel_label)

                    logit = logit.reshape(-1, 2)
                    pred = logit.argmax(dim=1)

                    intersection = (pred * voxel_label).sum()
                    union = torch.clamp(pred + voxel_label, max=1).sum()
                    if union > 0:
                        acc += float(intersection) / float(union)

                acc = acc/B
                loss = loss_in_out/B
                # loss_normal = (((normal_gt - normal_list)**2).sum(-1)+1e-6).sqrt().mean()
                # loss_normal = ((1-torch.matmul(normal_gt.unsqueeze(2), normal_list.unsqueeze(3)))).mean()
                return loss, acc

            self.loss = network_loss

    def adjust_learning_rate(self, optimizer, epoch):
        """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
        lr = self.config.learning_rate * (0.3 ** (epoch // 60))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


    @property
    def model_dir(self):
        return "{}_ae_{}".format(self.dataset_name, self.input_size)

    def load(self):
        # load previous checkpoint
        checkpoint_txt = str(os.path.join(self.checkpoint_path, "checkpoint"))
        print(checkpoint_txt, os.path.exists(checkpoint_txt), type(checkpoint_txt))
        if os.path.exists(checkpoint_txt):
            fin = open(checkpoint_txt)
            model_dir = fin.readline().strip()
            fin.close()
            self.bsp_network.load_state_dict(torch.load(model_dir), strict=True)
            print(" [*] Load SUCCESS")
            return True
        else:
            print(" [!] Load failed...")
            return False

    def save(self, epoch):
        if not os.path.exists(self.checkpoint_path):
            os.makedirs(self.checkpoint_path)
        save_dir = os.path.join(self.checkpoint_path,
                                self.checkpoint_name + str(self.sample_vox_size) + "-" + str(self.phase) + "-" + str(
                                    epoch) + ".pth")
        self.checkpoint_manager_pointer = (self.checkpoint_manager_pointer + 1) % self.max_to_keep
        # delete checkpoint
        if self.checkpoint_manager_list[self.checkpoint_manager_pointer] is not None:
            if os.path.exists(self.checkpoint_manager_list[self.checkpoint_manager_pointer]):
                os.remove(self.checkpoint_manager_list[self.checkpoint_manager_pointer])
        # save checkpoint
        torch.save(self.bsp_network.state_dict(), save_dir)
        # update checkpoint manager
        self.checkpoint_manager_list[self.checkpoint_manager_pointer] = save_dir
        # write file
        checkpoint_txt = os.path.join(self.checkpoint_path, "checkpoint")
        fout = open(checkpoint_txt, 'w')
        for i in range(self.max_to_keep):
            pointer = (self.checkpoint_manager_pointer + self.max_to_keep - i) % self.max_to_keep
            if self.checkpoint_manager_list[pointer] is not None:
                fout.write(self.checkpoint_manager_list[pointer] + "\n")
        fout.close()

    def train(self, config):
        # load previous checkpoint
        cls = self.config.dataset.split("/")[3].split("_")[1]
        writer = SummaryWriter("log/"+cls+"/pt")
        # self.load()
        shape_num = len(self.data_voxels)
        batch_index_list = np.arange(shape_num)

        print("\n\n----------net summary----------")
        print("training samples   ", shape_num)
        print("-------------------------------\n\n")

        start_time = time.time()
        assert config.epoch == 0 or config.iteration == 0
        training_epoch = config.epoch + int(config.iteration / shape_num)
        batch_num = int(shape_num / self.shape_batch_size)
        scheduler = CosineAnnealingLR(self.optimizer, training_epoch, eta_min=config.learning_rate*0.1)
        self.bsp_network.train()
        for epoch in range(0, training_epoch):
            scheduler.step()
            np.random.shuffle(batch_index_list)
            avg_loss_sp = 0
            avg_loss_tt = 0
            avg_loss_normal = 0
            avg_loss_d = 0
            avg_loss_r = 0
            avg_loss_z = 0
            avg_acc_z = 0
            avg_num = 0
            # self.adjust_learning_rate(self.optimizer, epoch)
            for idx in tqdm(range(batch_num)):
                dxb = batch_index_list[idx * self.shape_batch_size:(idx + 1) * self.shape_batch_size]
                batch_voxels = self.data_voxels[dxb].astype(np.float32)
                point_cloud = self.point_cloud[dxb].astype(np.float32)
                # F.conv2d(box, w, stride=2)
                # vertices, faces = mcubes.marching_cubes(batch_voxels[5, 0], 0.0)
                # mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
                # mesh.show()

                # batch_voxels = generate_softlabel(batch_voxels)
                batch_voxels = torch.from_numpy(batch_voxels)
                normal = torch.from_numpy(point_cloud)[:, :, 3:]
                point_cloud = torch.from_numpy(point_cloud)[:, :, 0:3]

                batch_voxels = batch_voxels.to(self.device)
                point_cloud = point_cloud.to(self.device)
                box_noise_scale = (point_cloud.max(1)[0]-point_cloud.min(1)[0]).max(-1)[0]


                target_point = point_cloud
                point_cloud += torch.randn_like(point_cloud)*0.005*box_noise_scale.unsqueeze(1).unsqueeze(1)
                # if self.data_voxels.shape[0] > 35000:
                #     point_cloud[:, :, [0, 2]] = point_cloud[:, :, [2, 0]]
                #     point_cloud[:, :, 2] = -point_cloud[:, :, 2]

                normal = normal.to(self.device)
                if np.random.random() > 0.5:
                    point_cloud = -1 * point_cloud
                    batch_voxels = torch.flip(batch_voxels, dims=[2, 3, 4])

                point_cloud += 0.5

                #  # ************************ erase ******************* #

                # # point_cloud -> voxel
                B = point_cloud.shape[0]

                # ss = batch_voxels.reshape(B, -1).sum(-1)
                point_index = (torch.clamp(point_cloud, min=0, max=0.99) * voxel_dim).long()
                index = (point_index[:, :, 0] * voxel_dim + point_index[:, :, 1]) * voxel_dim + point_index[:, :, 2]
                src = torch.ones_like(index).float()
                point_voxel = torch.zeros(B, voxel_dim**3, device=point_cloud.device).scatter_(1, index, src)
                point_voxel = point_voxel.reshape(B, 1, voxel_dim, voxel_dim, voxel_dim)

                expand_point = []
                expand_label = []
                for idx in range(B):
                    cloud = ndimage.binary_dilation(point_voxel[idx][0].cpu().numpy(),
                                                                 structure=ndimage.generate_binary_structure(3, 3),
                                                                 iterations=1)
                    cloud = torch.from_numpy(cloud).cuda()
                    cloud = torch.stack(torch.where(cloud == 1)).permute(1, 0)

                    dim = 16
                    box = (torch.cat([point_cloud[idx].max(dim=0)[0] + 2 / dim,
                                      point_cloud[idx].min(dim=0)[0] - 2 / dim], 0))
                    box = (torch.clamp(box, min=0, max=0.99) * dim).long().squeeze()
                    grid = torch.meshgrid(
                        [torch.arange(box[3], box[0]), torch.arange(box[4], box[1]), torch.arange(box[5], box[2])])
                    grid_point = torch.stack(grid).reshape(3, -1).permute(1, 0)
                    grid_point = grid_point.float() / dim + torch.rand(1, 3) / float(dim)


                    grid = (grid_point * voxel_dim).long().cuda()
                    cloud = torch.cat([cloud, grid], 0)
                    cloud = cloud[torch.randint(cloud.shape[0], (cloud.shape[0]//5,))]

                    label = batch_voxels[idx, 0, cloud[:,0], cloud[:,1], cloud[:,2]].long()
                    cloud = cloud.float() / voxel_dim+0.5/float(voxel_dim)
                    expand_point.append(cloud)
                    expand_label.append(label)

                self.bsp_network.zero_grad()

                rand_num = np.random.random()
                # if  rand_num < 0.5:
                #     rand_scale = 1

                rand_scale = 1
                # sqrdists = square_distance(point_cloud, point_cloud)
                # g_dis, group_idx = torch.topk(sqrdists, 5, dim=-1, largest=False, sorted=False)
                # scale = 1/(100*g_dis.mean(-1).mean(-1))
                # expand_point, point_cloud = [x * scale[idx]*rand_scale for idx, x in enumerate(expand_point)], [x * scale[idx]*rand_scale for idx, x in enumerate(point_cloud)]
                # point_cloud = torch.stack(point_cloud)

                loss, acc = self.loss(expand_label, expand_point, point_cloud,
                                                              self.bsp_network.local_decoder, target_point)
                loss.backward()
                self.optimizer.step()
                avg_loss_sp += loss.item()
                avg_loss_tt += acc
                avg_loss_normal += 0
                # avg_loss_d += err_D.item()
                # avg_loss_r += err_R.item()
                # avg_loss_z += err_Z.item()
                # avg_acc_z += acc_Z.item()
                avg_num += 1
            writer.add_scalar("loss", avg_loss_sp / avg_num, global_step=epoch)
            writer.add_scalar("acc", avg_loss_tt / avg_num, global_step=epoch)
            print(str(
                self.sample_vox_size) + " Epoch: [%2d/%2d] time: %4.4f, loss: %.6f, iou: %.6f" % (
                      epoch, training_epoch, time.time() - start_time, avg_loss_sp / avg_num, avg_loss_tt / avg_num))
            # if epoch%10==9:
            # 	self.test_1(config,"train_"+str(self.sample_vox_size)+"_"+str(epoch))
            if epoch % 10 == 9:
                self.save(epoch)

        self.save(training_epoch)

    def test_1(self, config, name):
        multiplier = int(self.real_size / self.test_size)
        multiplier2 = multiplier * multiplier

        if config.phase == 0:
            thres = 0.5
        else:
            thres = 0.99

        t = np.random.randint(len(self.data_voxels))
        model_float = np.zeros([self.real_size + 2, self.real_size + 2, self.real_size + 2], np.float32)
        batch_voxels = self.data_voxels[t:t + 1].astype(np.float32)
        batch_voxels = torch.from_numpy(batch_voxels)
        batch_voxels = batch_voxels.to(self.device)
        _, out_m, _, _ = self.bsp_network(batch_voxels, None, None, None, is_training=False)
        for i in range(multiplier):
            for j in range(multiplier):
                for k in range(multiplier):
                    minib = i * multiplier2 + j * multiplier + k
                    point_coord = self.coords[minib:minib + 1]
                    _, _, _, net_out = self.bsp_network(None, None, out_m, point_coord, is_training=False)
                    if config.phase != 0:
                        net_out = torch.clamp(1 - net_out, min=0, max=1)
                    model_float[self.aux_x + i + 1, self.aux_y + j + 1, self.aux_z + k + 1] = np.reshape(
                        net_out.detach().cpu().numpy(), [self.test_size, self.test_size, self.test_size])

        vertices, triangles = mcubes.marching_cubes(model_float, thres)
        vertices = (vertices - 0.5) / self.real_size - 0.5
        # output ply sum
        write_ply_triangle(config.sample_dir + "/" + name + ".ply", vertices, triangles)
        print("[sample]")

    # output bsp shape as ply
    def test_bsp(self, config):
        # load previous checkpoint
        if not self.load(): exit(-1)

        w2 = self.bsp_network.generator.convex_layer_weights.detach().cpu().numpy()

        dima = self.test_size
        dim = self.real_size
        multiplier = int(dim / dima)
        multiplier2 = multiplier * multiplier

        self.bsp_network.eval()
        for t in range(config.start, min(len(self.data_voxels), config.end)):
            model_float = np.ones([self.real_size, self.real_size, self.real_size, self.c_dim], np.float32)
            batch_voxels = self.data_voxels[t:t + 1].astype(np.float32)
            batch_voxels = torch.from_numpy(batch_voxels)
            batch_voxels = batch_voxels.to(self.device)
            _, out_m, _, _ = self.bsp_network(batch_voxels, None, None, None, is_training=False)
            for i in range(multiplier):
                for j in range(multiplier):
                    for k in range(multiplier):
                        minib = i * multiplier2 + j * multiplier + k
                        point_coord = self.coords[minib:minib + 1]
                        _, _, model_out, _ = self.bsp_network(None, None, out_m, point_coord, is_training=False)
                        model_float[self.aux_x + i, self.aux_y + j, self.aux_z + k, :] = np.reshape(
                            model_out.detach().cpu().numpy(),
                            [self.test_size, self.test_size, self.test_size, self.c_dim])

            out_m = out_m.detach().cpu().numpy()

            bsp_convex_list = []
            model_float = model_float < 0.01
            model_float_sum = np.sum(model_float, axis=3)
            for i in range(self.c_dim):
                slice_i = model_float[:, :, :, i]
                if np.max(slice_i) > 0:  # if one voxel is inside a convex
                    if np.min(
                            model_float_sum - slice_i * 2) >= 0:  # if this convex is redundant, i.e. the convex is inside the shape
                        model_float_sum = model_float_sum - slice_i
                    else:
                        box = []
                        for j in range(self.p_dim):
                            if w2[j, i] > 0.01:
                                a = -out_m[0, 0, j]
                                b = -out_m[0, 1, j]
                                c = -out_m[0, 2, j]
                                d = -out_m[0, 3, j]
                                box.append([a, b, c, d])
                        if len(box) > 0:
                            bsp_convex_list.append(np.array(box, np.float32))

            # print(bsp_convex_list)
            print(len(bsp_convex_list))

            # convert bspt to mesh
            # vertices, polygons = get_mesh(bsp_convex_list)
            # use the following alternative to merge nearby vertices to get watertight meshes
            vertices, polygons = get_mesh_watertight(bsp_convex_list)

            # output ply
            write_ply_polygon(config.sample_dir + "/" + str(t) + "_bsp.ply", vertices, polygons)

    # output bsp shape as ply and point cloud as ply
    def test_mesh_point(self, config):
        # load previous checkpoint
        if not self.load(): exit(-1)

        w2 = self.bsp_network.generator.convex_layer_weights.detach().cpu().numpy()
        dima = self.test_size
        dim = self.real_size
        multiplier = int(dim / dima)
        multiplier2 = multiplier * multiplier

        self.bsp_network.eval()
        for t in range(config.start, min(len(self.data_voxels), config.end)):
            print(t)
            model_float = np.ones([self.real_size, self.real_size, self.real_size, self.c_dim], np.float32)
            model_float_combined = np.ones([self.real_size, self.real_size, self.real_size], np.float32)
            batch_voxels = self.data_voxels[t:t + 1].astype(np.float32)
            batch_voxels = torch.from_numpy(batch_voxels)
            batch_voxels = batch_voxels.to(self.device)
            _, out_m, _, _ = self.bsp_network(batch_voxels, None, None, None, is_training=False)
            for i in range(multiplier):
                for j in range(multiplier):
                    for k in range(multiplier):
                        minib = i * multiplier2 + j * multiplier + k
                        point_coord = self.coords[minib:minib + 1]
                        _, _, model_out, model_out_combined = self.bsp_network(None, None, out_m, point_coord,
                                                                               is_training=False)
                        model_float[self.aux_x + i, self.aux_y + j, self.aux_z + k, :] = np.reshape(
                            model_out.detach().cpu().numpy(),
                            [self.test_size, self.test_size, self.test_size, self.c_dim])
                        model_float_combined[self.aux_x + i, self.aux_y + j, self.aux_z + k] = np.reshape(
                            model_out_combined.detach().cpu().numpy(), [self.test_size, self.test_size, self.test_size])

            out_m_ = out_m.detach().cpu().numpy()

            # whether to use post processing to remove convexes that are inside the shape
            post_processing_flag = False

            if post_processing_flag:
                bsp_convex_list = []
                model_float = model_float < 0.01
                model_float_sum = np.sum(model_float, axis=3)
                unused_convex = np.ones([self.c_dim], np.float32)
                for i in range(self.c_dim):
                    slice_i = model_float[:, :, :, i]
                    if np.max(slice_i) > 0:  # if one voxel is inside a convex
                        if np.min(
                                model_float_sum - slice_i * 2) >= 0:  # if this convex is redundant, i.e. the convex is inside the shape
                            model_float_sum = model_float_sum - slice_i
                        else:
                            box = []
                            for j in range(self.p_dim):
                                if w2[j, i] > 0.01:
                                    a = -out_m_[0, 0, j]
                                    b = -out_m_[0, 1, j]
                                    c = -out_m_[0, 2, j]
                                    d = -out_m_[0, 3, j]
                                    box.append([a, b, c, d])
                            if len(box) > 0:
                                bsp_convex_list.append(np.array(box, np.float32))
                                unused_convex[i] = 0

                # convert bspt to mesh
                # vertices, polygons = get_mesh(bsp_convex_list)
                # use the following alternative to merge nearby vertices to get watertight meshes
                vertices, polygons = get_mesh_watertight(bsp_convex_list)

                # output ply
                write_ply_polygon(config.sample_dir + "/" + str(t) + "_bsp.ply", vertices, polygons)
                # output obj
                # write_obj_polygon(config.sample_dir+"/"+str(t)+"_bsp.obj", vertices, polygons)

                # sample surface points
                sampled_points_normals = sample_points_polygon(vertices, polygons, 16384)
                # check point inside shape or not
                point_coord = np.reshape(sampled_points_normals[:, :3] + sampled_points_normals[:, 3:] * 1e-4,
                                         [1, -1, 3])
                point_coord = np.concatenate([point_coord, np.ones([1, point_coord.shape[1], 1], np.float32)], axis=2)
                _, _, _, sample_points_value = self.bsp_network(None, None, out_m,
                                                                torch.from_numpy(point_coord).to(self.device),
                                                                convex_mask=torch.from_numpy(
                                                                    np.reshape(unused_convex, [1, 1, -1])).to(
                                                                    self.device), is_training=False)
                sample_points_value = sample_points_value.detach().cpu().numpy()
                sampled_points_normals = sampled_points_normals[sample_points_value[0, :, 0] > 1e-4]
                print(len(bsp_convex_list), len(sampled_points_normals))
                np.random.shuffle(sampled_points_normals)
                write_ply_point_normal(config.sample_dir + "/" + str(t) + "_pc.ply", sampled_points_normals[:4096])
            else:
                bsp_convex_list = []
                model_float = model_float < 0.01
                model_float_sum = np.sum(model_float, axis=3)
                for i in range(self.c_dim):
                    slice_i = model_float[:, :, :, i]
                    if np.max(slice_i) > 0:  # if one voxel is inside a convex
                        # if np.min(model_float_sum-slice_i*2)>=0: #if this convex is redundant, i.e. the convex is inside the shape
                        #	model_float_sum = model_float_sum-slice_i
                        # else:
                        box = []
                        for j in range(self.p_dim):
                            if w2[j, i] > 0.01:
                                a = -out_m_[0, 0, j]
                                b = -out_m_[0, 1, j]
                                c = -out_m_[0, 2, j]
                                d = -out_m_[0, 3, j]
                                box.append([a, b, c, d])
                        if len(box) > 0:
                            bsp_convex_list.append(np.array(box, np.float32))

                # convert bspt to mesh
                # vertices, polygons = get_mesh(bsp_convex_list)
                # use the following alternative to merge nearby vertices to get watertight meshes
                vertices, polygons = get_mesh_watertight(bsp_convex_list)

                # output ply
                write_ply_polygon(config.sample_dir + "/" + str(t) + "_bsp.ply", vertices, polygons)
                # output obj
                # write_obj_polygon(config.sample_dir+"/"+str(t)+"_bsp.obj", vertices, polygons)

                # sample surface points
                sampled_points_normals = sample_points_polygon_vox64(vertices, polygons, model_float_combined, 16384)
                # check point inside shape or not
                point_coord = np.reshape(sampled_points_normals[:, :3] + sampled_points_normals[:, 3:] * 1e-4,
                                         [1, -1, 3])
                point_coord = np.concatenate([point_coord, np.ones([1, point_coord.shape[1], 1], np.float32)], axis=2)
                _, _, _, sample_points_value = self.bsp_network(None, None, out_m,
                                                                torch.from_numpy(point_coord).to(self.device),
                                                                is_training=False)
                sample_points_value = sample_points_value.detach().cpu().numpy()
                sampled_points_normals = sampled_points_normals[sample_points_value[0, :, 0] > 1e-4]
                print(len(bsp_convex_list), len(sampled_points_normals))
                np.random.shuffle(sampled_points_normals)
                write_ply_point_normal(config.sample_dir + "/" + str(t) + "_pc.ply", sampled_points_normals[:4096])

    # output bsp shape as obj with color
    def test_mesh_obj_material(self, config):
        # load previous checkpoint
        if not self.load(): exit(-1)

        w2 = self.bsp_network.generator.convex_layer_weights.detach().cpu().numpy()

        dima = self.test_size
        dim = self.real_size
        multiplier = int(dim / dima)
        multiplier2 = multiplier * multiplier

        # write material
        # all output shapes share the same material
        # which means the same convex always has the same color for different shapes
        # change the colors in default.mtl to visualize correspondences between shapes
        fout2 = open(config.sample_dir + "/default.mtl", 'w')
        for i in range(self.c_dim):
            fout2.write("newmtl m" + str(i + 1) + "\n")  # material id
            fout2.write("Kd 0.80 0.80 0.80\n")  # color (diffuse) RGB 0.00-1.00
            fout2.write("Ka 0 0 0\n")  # color (ambient) leave 0s
        fout2.close()

        self.bsp_network.eval()
        for t in range(config.start, min(len(self.data_voxels), config.end)):
            model_float = np.ones([self.real_size, self.real_size, self.real_size, self.c_dim], np.float32)
            batch_voxels = self.data_voxels[t:t + 1].astype(np.float32)
            batch_voxels = torch.from_numpy(batch_voxels)
            batch_voxels = batch_voxels.to(self.device)
            _, out_m, _, _ = self.bsp_network(batch_voxels, None, None, None, is_training=False)
            for i in range(multiplier):
                for j in range(multiplier):
                    for k in range(multiplier):
                        minib = i * multiplier2 + j * multiplier + k
                        point_coord = self.coords[minib:minib + 1]
                        _, _, model_out, _ = self.bsp_network(None, None, out_m, point_coord, is_training=False)
                        model_float[self.aux_x + i, self.aux_y + j, self.aux_z + k, :] = np.reshape(
                            model_out.detach().cpu().numpy(),
                            [self.test_size, self.test_size, self.test_size, self.c_dim])

            out_m = out_m.detach().cpu().numpy()

            bsp_convex_list = []
            color_idx_list = []
            model_float = model_float < 0.01
            model_float_sum = np.sum(model_float, axis=3)
            for i in range(self.c_dim):
                slice_i = model_float[:, :, :, i]
                if np.max(slice_i) > 0:  # if one voxel is inside a convex
                    if np.min(
                            model_float_sum - slice_i * 2) >= 0:  # if this convex is redundant, i.e. the convex is inside the shape
                        model_float_sum = model_float_sum - slice_i
                    else:
                        box = []
                        for j in range(self.p_dim):
                            if w2[j, i] > 0.01:
                                a = -out_m[0, 0, j]
                                b = -out_m[0, 1, j]
                                c = -out_m[0, 2, j]
                                d = -out_m[0, 3, j]
                                box.append([a, b, c, d])
                        if len(box) > 0:
                            bsp_convex_list.append(np.array(box, np.float32))
                            color_idx_list.append(i)

            # print(bsp_convex_list)
            print(len(bsp_convex_list))

            # convert bspt to mesh
            vertices = []

            # write obj
            fout2 = open(config.sample_dir + "/" + str(t) + "_bsp.obj", 'w')
            fout2.write("mtllib default.mtl\n")

            for i in range(len(bsp_convex_list)):
                vg, tg = get_mesh([bsp_convex_list[i]])
                vbias = len(vertices) + 1
                vertices = vertices + vg

                fout2.write("usemtl m" + str(color_idx_list[i] + 1) + "\n")
                for ii in range(len(vg)):
                    fout2.write("v " + str(vg[ii][0]) + " " + str(vg[ii][1]) + " " + str(vg[ii][2]) + "\n")
                for ii in range(len(tg)):
                    fout2.write("f")
                    for jj in range(len(tg[ii])):
                        fout2.write(" " + str(tg[ii][jj] + vbias))
                    fout2.write("\n")

            fout2.close()


    # output h3
    def test_dae3(self, config):
        # load previous checkpoint
        if not self.load(): exit(-1)
        self.bsp_network.train()
        acc_mean = 0
        acc2_mean = 0
        total_num = 0
        acc_list = []
        data_num = min(len(self.data_voxels), config.end) - config.start
        with torch.no_grad():
            for t in range(config.start, min(len(self.data_voxels), config.end)):
                batch_voxels_ = self.data_voxels[t:t + 1].astype(np.float32)
                batch_voxels = torch.from_numpy(batch_voxels_)
                batch_voxels = batch_voxels.to(self.device)

                point_cloud = self.point_cloud[t:t + 1].astype(np.float32)
                normal = torch.from_numpy(point_cloud)[:, :, 3:]
                point_cloud = torch.from_numpy(point_cloud[:, :, 0:3])
                point_cloud = point_cloud.to(self.device)
                # if self.data_voxels.shape[0] > 8700:
                #     point_cloud[:, :, [0,2]] = point_cloud[:, :, [2,0]]
                #     point_cloud[:, :, 2] = -point_cloud[:, :, 2]

                # point_cloud[:,[0,2]]= point_cloud[:,[2,0]]
                # if np.random.random() > 0.5:
                #     point_cloud = -1 * point_cloud
                #     batch_voxels = torch.flip(batch_voxels, dims=[2, 3, 4])
                normal = normal.to(self.device)

                box_noise_scale = (point_cloud.max(1)[0] - point_cloud.min(1)[0]).max(-1)[0]

                point_cloud += torch.randn_like(point_cloud) * 0.005 * box_noise_scale.unsqueeze(1).unsqueeze(1)

                point_cloud = point_cloud + 0.5
                # batch_voxels = torch.flip(batch_voxels, dims=[4])

                ## ************************ erase ******************* #
                inputs = torch.zeros_like(batch_voxels).copy_(batch_voxels)

                # ## ************************ erase *******************#
                    # # point_cloud -> voxel
                B = point_cloud.shape[0]
                point_index = (torch.clamp(point_cloud, min=0, max=0.99) * voxel_dim).long()
                index = (point_index[:, :, 0] * voxel_dim + point_index[:, :, 1]) * voxel_dim + point_index[:, :, 2]
                src = torch.ones_like(index).float()
                point_voxel = torch.zeros(B, voxel_dim ** 3, device=point_cloud.device).scatter_(1, index, src)
                point_voxel = point_voxel.reshape(B, 1, voxel_dim, voxel_dim, voxel_dim)

                expand_point = []
                expand_label = []
                for idx in range(B):
                    box = (torch.cat([point_cloud.max(dim=1)[0]+2/voxel_dim,
                           point_cloud.min(dim=1)[0]-2/voxel_dim], 1))
                    box = (torch.clamp(box, min=0, max=0.99)*voxel_dim).long().squeeze()
                    grid = torch.meshgrid([torch.arange(box[3], box[0]), torch.arange(box[4], box[1]), torch.arange(box[5], box[2])])
                    label = batch_voxels[idx, 0, grid[0].reshape(-1), grid[1].reshape(-1), grid[2].reshape(-1)].long()
                    grid_point = torch.stack(grid).reshape(3, -1).permute(1, 0).cuda()
                    grid_point = grid_point.float()/voxel_dim+0.5/float(voxel_dim)
                    expand_point.append(grid_point)
                    expand_label.append(label)

                    # cloud = ndimage.binary_dilation(point_voxel[idx][0].cpu().numpy(),
                    #                                 structure=ndimage.generate_binary_structure(3, 3),
                    #                                 iterations=1)
                    # cloud = torch.from_numpy(cloud).cuda()
                    # cloud = torch.stack(torch.where(cloud == 1)).permute(1, 0)
                    # label = batch_voxels[idx, 0, cloud[:, 0], cloud[:, 1], cloud[:, 2]].long()
                    # cloud = cloud.float() / voxel_dim
                    # expand_point.append(cloud+0.5/float(voxel_dim))
                    # expand_label.append(label)

                sqrdists = square_distance(point_cloud, point_cloud)
                g_dis, group_idx = torch.topk(sqrdists, 5, dim=-1, largest=False, sorted=False)
                scale = 1 / (100 * g_dis.mean(-1).mean(-1))
                expand_point, point_cloud = [x * scale[idx]  for idx, x in enumerate(expand_point)], [
                    x * scale[idx] for idx, x in enumerate(point_cloud)]
                point_cloud = torch.stack(point_cloud)

                net_out_all, acc, acc2 = self.bsp_network(expand_label, expand_point, point_cloud, point_voxel, scale, is_training=False)
                net_out_all = net_out_all.squeeze()
                acc_list.append(acc2)
                acc_mean += acc
                acc2_mean += acc2
                total_num += 1
                point_cloud = [x / scale[idx] for idx, x in enumerate(point_cloud)]
                # net_out_all = smooth_process(net_out_all)

                # idx = torch.stack(torch.where(net_out_all == 1)).permute(1, 0)
                # idx2 = torch.stack(torch.where(batch_voxels[0, 0, :, :, :] == 1)).permute(1, 0)
                # np.savetxt("points/pc" + str(0) + ".txt", point_cloud[0].float().detach().cpu().numpy()-0.5)
                # np.savetxt("points/bv_" + str(0) + ".txt", idx2.float().detach().cpu().numpy() / 64.0 - 0.5 + 0.5 / 64)
                # np.savetxt("points/net_out_all_" + str(0) + ".txt", idx.float().detach().cpu().numpy() / 64.0- 0.5 + 0.5 / 64)
                # exit(0)
                net_out_all = net_out_all.squeeze()
                net_out_all = F.pad(net_out_all, (1, 1, 1, 1, 1, 1), 'constant', 0)
                net_out_all = net_out_all.cpu().detach().numpy()
                # pred = net_out_all
                # voxel_label = batch_voxels_[0, 0, :, :, :].astype(np.int)
                # intersection = (pred * voxel_label).sum()
                # union = np.clip(pred + voxel_label, a_min=0, a_max=1).sum()
                # sss = float(intersection) / float(union)

                vertices, triangles = mcubes.marching_cubes(net_out_all, 0.5)
                vertices = (vertices - 1 + 0.5) / float(voxel_dim) - 0.5
                # output prediction
                write_ply_triangle(config.sample_dir + "/" + str(t) + "_vox.ply", vertices, triangles)
                mesh = o3d.io.read_triangle_mesh(config.sample_dir + "/" + str(t) + "_vox.ply")
                mesh = mesh.filter_smooth_laplacian(number_of_iterations=3)
                vertices, polygons = np.asarray(mesh.vertices), np.asarray(mesh.triangles)
                write_ply_triangle(config.sample_dir + "/" + str(t) + "_vox.ply", vertices, triangles)

                # point_cloud[:, :, [0, 2]] = point_cloud[:, :, [2, 0]]
                np.savetxt(config.sample_dir + "/" + str(t) + "_pc.txt", (point_cloud[0]-0.5).cpu().detach().numpy())
                # batch_voxels_ = torch.flip(batch_voxels_, dims=[4])
                # active_grids_large = ndimage.binary_dilation(batch_voxels_[0, 0, :, :, :],
                #                                              structure=ndimage.generate_binary_structure(3, 3),
                #                                              iterations=1)
                batch_voxels_ = batch_voxels[0, 0, :, :, :].squeeze()
                batch_voxels_ = F.pad(batch_voxels_, (1, 1, 1, 1, 1, 1), 'constant', 0)
                batch_voxels_ = batch_voxels_.detach().cpu().numpy()
                vertices, triangles = mcubes.marching_cubes(batch_voxels_, 0.5)
                vertices = (vertices -1 + 0.5) / 64 - 0.5
                # output ground truth
                write_ply_triangle(config.sample_dir + "/" + str(t) + "_gt.ply", vertices, triangles)
                mesh = o3d.io.read_triangle_mesh(config.sample_dir + "/" + str(t) + "_gt.ply")
                mesh = mesh.filter_smooth_laplacian(number_of_iterations=3)
                vertices, polygons = np.asarray(mesh.vertices), np.asarray(mesh.triangles)
                write_ply_triangle(config.sample_dir + "/" + str(t) + "_gt.ply", vertices, triangles)

                print("[sample%d]"%t, acc)
        print(torch.topk(torch.tensor(acc_list), k=min(100, data_num), largest=True)[1])
        print("box_predict acc is： ", acc_mean/total_num, ", voxel acc is： ", acc2_mean/total_num)

    def get_z(self, config):
        # load previous checkpoint
        if not self.load(): exit(-1)
        last = os.path.split(self.dataset_name)[1]
        hdf5_path = self.checkpoint_dir + '/' + self.model_dir + '/' + last + '_train_z.hdf5'
        # print(self.checkpoint_dir,",", self.model_dir, ",", self.dataset_name, last)
        shape_num = len(self.data_voxels)
        hdf5_file = h5py.File(hdf5_path, mode='w')
        hdf5_file.create_dataset("zs", [shape_num, self.ef_dim * 8], np.float32)

        self.bsp_network.eval()
        print(shape_num)
        for t in range(shape_num):
            batch_voxels = self.data_voxels[t:t + 1].astype(np.float32)
            batch_voxels = torch.from_numpy(batch_voxels)
            batch_voxels = batch_voxels.to(self.device)
            out_z, _, _, _ = self.bsp_network(batch_voxels, None, None, None, is_training=False)
            hdf5_file["zs"][t:t + 1, :] = out_z.detach().cpu().numpy()

        hdf5_file.close()
        print("[z]")


def smooth_process(batch_voxels):
    batch_voxels_ = F.pad(batch_voxels, (1, 1, 1, 1, 1, 1), 'constant', 0)
    result = torch.zeros_like(batch_voxels)
    for x_shift in range(0,3):
        for y_shift in range(0, 3):
            for z_shift in range(0, 3):
                result += batch_voxels_[x_shift:x_shift+64, y_shift:y_shift+64, z_shift:z_shift+64]
    batch_voxels[torch.where(result == 1)] = 0
    batch_voxels[torch.where(result // 16 == 1)] = 1
    return batch_voxels
