import os
import time
import math
import random

import numpy
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
from attention_layer import local_decoder, seg_decoder
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

voxel_dim = 128
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
        self.local_decoder = seg_decoder(self.ef_dim) # local_decoder(self.ef_dim)
        # self.zero_decoder = zero_decoder(ef_dim)

    def forward(self, expand_label, expand_point, point_cloud, point_voxel, scale, is_training=False):
        if not is_training:
            query_list = torch.chunk(expand_point, 30, dim=1)
            output_list = []
            for q in query_list:
                o, _ = self.local_decoder(point_cloud, q, scale)
                output_list.append(torch.stack(o))
            output = torch.cat(output_list, dim=1)
            B = len(output)

            result = torch.zeros(B, voxel_dim ** 3, dtype=torch.float32, device=point_cloud.device)  # .scatter_(1, index, src)

            result = result.reshape(B, 1, voxel_dim, voxel_dim, voxel_dim)
            result = result.copy_(point_voxel)
            result = result.reshape(B, voxel_dim, voxel_dim, voxel_dim)

            loss_in_out = 0
            acc = 0
            for idx in range(B):
                cloud = expand_point[idx]+0.5
                voxel_position = (cloud/scale[idx]*voxel_dim).long()
                logit = F.log_softmax(output[idx], dim=1)
                voxel_label = expand_label[idx].reshape(-1).long()

                logit = logit.reshape(-1, 2)
                pred = logit[:, 1]
                # pred = logit.argmax(dim=1)
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
        if not (config.train):
            self.dataset_load = self.dataset_name + '_test'
        self.checkpoint_dir = config.checkpoint_dir
        self.checkpoint_path = os.path.join(self.checkpoint_dir, self.dataset_name)
        self.data_dir = config.data_dir

        cls_str = os.path.split(os.path.split(self.dataset_load)[0])[1]
        cls = cls_str.split("_")[0]

        # self.root_data_dir = "/disk2/occupancy_networks-master/data/ShapeNet"
        # self.root_data_dir = "/disk2/occupancy_networks-master/data/ABC"
        # self.root_data_dir = "/disk2/occupancy_networks-master/data/ABC_convert"
        if cls == "001":
            self.root_data_dir = "/disk2/occupancy_networks-master/data/ABC"
        else:
            self.root_data_dir = "/disk2/occupancy_networks-master/data/ShapeNet"

        if cls == '00000000':
            data_list = []
            for ff in os.listdir(self.root_data_dir):
                if config.train:
                    data_txt_name = os.path.join(self.root_data_dir, ff, "train.lst")
                else:
                    data_txt_name = os.path.join(self.root_data_dir, ff, "test.lst")
                # data_txt_name = os.path.join(self.root_data_dir, cls, "train.lst")
                if os.path.exists(data_txt_name):
                    data_dict = open(data_txt_name, 'r')
                    txt_data = data_dict.read().split("\n")[0:-1]
                    txt_data = [ff + "/" + x for x in txt_data]
                    data_list += txt_data
                else:
                    print("error: cannot load " + data_txt_name)
                    exit(0)
            self.txt_data = data_list

        else:
            if config.train:
                data_txt_name = os.path.join(self.root_data_dir, cls, "train.lst")
            else:
                data_txt_name = os.path.join(self.root_data_dir, cls, "test.lst")
            # data_txt_name = os.path.join(self.root_data_dir, cls, "train.lst")
            if os.path.exists(data_txt_name):
                data_dict = open(data_txt_name, 'r')
                self.txt_data = data_dict.read().split("\n")[0:-1]
                self.txt_data = [cls+"/"+x for x in self.txt_data]
            else:
                print("error: cannot load " + data_txt_name)
                exit(0)

        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            torch.backends.cudnn.benchmark = True
        else:
            self.device = torch.device('cpu')

        # build model
        # self.denoise_net = denoise_net()
        # self.denoise_net.to(self.device)
        # self.load(task='denoise')

        self.bsp_network = bsp_network(self.phase, 32)
        self.bsp_network.to(self.device)
        self.optimizer = torch.optim.Adam(self.bsp_network.parameters(), lr=config.learning_rate,
                                          betas=(config.beta1, 0.999))

        # pytorch does not have a checkpoint manager
        # have to define it myself to manage max num of checkpoints to keep
        self.max_to_keep = 2
        self.checkpoint_path = os.path.join(self.checkpoint_dir, self.dataset_name)
        self.checkpoint_name = 'BSP_AE.model'
        self.checkpoint_manager_list = [None] * self.max_to_keep
        self.checkpoint_manager_pointer = 0

        if config.phase == 1:
            # phase 1 hard discrete for bsp
            # loss_sp
            def network_loss(expand_label, expand_point, point_cloud, cls, query_s):
                output, weight = cls(point_cloud, expand_point, query_s)
                # flop, para = profile(cls, inputs=(point_cloud, expand_point, None))
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

                acc = acc / B
                loss = loss_in_out / B

                # loss_normal = (((normal_gt - normal_list)**2).sum(-1)+1e-6).sqrt().mean()
                # loss_normal = ((1-torch.matmul(normal_gt.unsqueeze(2), normal_list.unsqueeze(3)))).mean()
                return loss, acc, loss

            self.loss = network_loss

    def adjust_learning_rate(self, optimizer, epoch):
        """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
        lr = self.config.learning_rate * (0.3 ** (epoch // 60))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    def load(self, task=''):
        # load previous checkpoint
        checkpoint_txt = str(os.path.join(self.checkpoint_path, "checkpoint"+task))
        print(checkpoint_txt, os.path.exists(checkpoint_txt), type(checkpoint_txt))
        if os.path.exists(checkpoint_txt):
            fin = open(checkpoint_txt)
            model_dir = fin.readline().strip()
            fin.close()
            if task == 'denoise':
                self.denoise_net.load_state_dict(torch.load(model_dir), strict=True)
                print(" [*] Load Denoise SUCCESS")
            else:
                state  = torch.load(model_dir)
                self.bsp_network.load_state_dict(torch.load(model_dir), strict=True)
                print(" [*] Load SUCCESS")

            return True
        else:
            print(" [!] Load failed...")
            return False

    def save(self, epoch, task=''):
        if not os.path.exists(self.checkpoint_path):
            os.makedirs(self.checkpoint_path)
        save_dir = os.path.join(self.checkpoint_path,
                                task + str(self.sample_vox_size) + "-" + str(self.phase) + "-" + str(
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
        checkpoint_txt = os.path.join(self.checkpoint_path, "checkpoint"+task)
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
        shape_num = len(self.txt_data)
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
            avg_loss_chamfer = 0
            avg_loss_d = 0
            avg_loss_r = 0
            avg_loss_z = 0
            avg_acc_z = 0
            avg_num = 0
            # self.save(epoch, task='denoise')
            # exit(0)
            # self.adjust_learning_rate(self.optimizer, epoch)
            for idx in tqdm(range(batch_num)):
                dxb = batch_index_list[idx * self.shape_batch_size:(idx + 1) * self.shape_batch_size].tolist()
                # point_cloud, query_p, query_l, query_s, points_chamfer = ABC_load_util(self.txt_data, dxb, self.root_data_dir)

                point_cloud, query_p, query_l, _, _ = load_data(self.txt_data, dxb, self.root_data_dir)
                # point_cloud = clean_point
                # np.savetxt("pc3000.txt", clean_point[0].numpy(), delimiter=";")
                # # # np.savetxt("pc10000.txt", points_chamfer[0].numpy(), delimiter=";")
                # np.savetxt("gt.txt", query_p[0][np.where(query_l[0]>0.5)], delimiter=";")
                # exit(0)
                random_padding = 1.1  # (np.random.random()-0.5)*0.25 + 1
                query_p = query_p.to(self.device)/random_padding
                query_l = query_l.to(self.device)
                # query_s = query_s.to(self.device)/random_padding
                point_cloud = point_cloud.to(self.device)/random_padding
                self.bsp_network.zero_grad()
                loss_occ, acc, loss_sdf = self.loss(query_l, query_p, point_cloud,
                                      self.bsp_network.local_decoder, point_cloud)
                # loss, chamfer, acc = self.bsp_network.network_loss(query_l, query_p, point_cloud, clean_point, points_chamfer)
                loss = loss_occ # + loss_sdf
                loss.backward()
                self.optimizer.step()
                avg_loss_sp += loss_sdf.item()
                avg_loss_d += loss_occ.item()
                avg_loss_tt += acc
                # avg_loss_chamfer += chamfer.item()
                # avg_loss_d += err_D.item()
                # avg_loss_r += err_R.item()
                # avg_loss_z += err_Z.item()
                # avg_acc_z += acc_Z.item()
                avg_num += 1
            writer.add_scalar("loss", avg_loss_sp / avg_num, global_step=epoch)
            writer.add_scalar("acc", avg_loss_tt / avg_num, global_step=epoch)
            print(str(
                self.sample_vox_size) + " Epoch: [%2d/%2d] time: %4.4f, loss_sdf: %.6f, loss_occ: %.6f, iou: %.6f" % (
                      epoch, training_epoch, time.time() - start_time, avg_loss_sp / avg_num, avg_loss_d / avg_num, avg_loss_tt / avg_num))
            # if epoch%10==9:
            # 	self.test_1(config,"train_"+str(self.sample_vox_size)+"_"+str(epoch))
            if epoch % 10 == 9:
                self.save(epoch)

        self.save(training_epoch)


    def test_dae3(self, config):
        # load previous checkpoint
        result_path = config.sample_dir + "/" + self.txt_data[0].split("/")[0]+"/"
        if not os.path.exists(result_path):
            os.mkdir(result_path)
        if not self.load(): exit(-1)
        self.bsp_network.train()
        acc_mean = 0
        acc2_mean = 0
        total_num = 0
        acc_list = []
        data_num = min(len(self.txt_data), config.end) - config.start
        padding=1.1
        total_time = 0
        with torch.no_grad():
            for t in range(config.start, min(len(self.txt_data), config.end)):
                filename = self.txt_data[t:t + 1]

                point_cloud = load_data_test(self.txt_data, [t], self.root_data_dir)
                # point_cloud = clean_point
                # clean_point = point_cloud
                point_cloud = point_cloud.to(self.device)/padding
                # clean_point = clean_point.to(self.device)/padding
                point_cloud = point_cloud + 0.5
                ## ************************ erase ******************* #

                B = point_cloud.shape[0]
                point_index = (torch.clamp(point_cloud, min=0, max=0.99) * voxel_dim).long()
                index = (point_index[:, :, 0] * voxel_dim + point_index[:, :, 1]) * voxel_dim + point_index[:, :, 2]
                src = torch.ones_like(index).float()
                point_voxel = torch.ones(B, voxel_dim ** 3, device=point_cloud.device)  # .scatter_(1, index, src)
                point_voxel = point_voxel.reshape(B, 1, voxel_dim, voxel_dim, voxel_dim)*-100

                B = point_cloud.shape[0]
                expand_point = []
                expand_label = []
                point_cloud = point_cloud
                for idx in range(B):
                    box = (torch.cat([point_cloud.max(dim=1)[0]+0.05,
                           point_cloud.min(dim=1)[0]-0.05], 1))
                    box = torch.clamp((box*voxel_dim).round().long(), min=0, max=voxel_dim).squeeze()
                    grid = torch.meshgrid([torch.arange(box[3], box[0]), torch.arange(box[4], box[1]), torch.arange(box[5], box[2])])
                    # label = batch_voxels[grid[0].reshape(-1), grid[1].reshape(-1), grid[2].reshape(-1)]
                    # label = torch.zeros(test_dim**3, dtype=torch.long)# batch_voxels[idx, 0, grid[0].reshape(-1), grid[1].reshape(-1), grid[2].reshape(-1)].long()
                    grid = torch.stack(grid).reshape(3, -1).permute(1, 0).cuda()
                    grid_point = grid.float()/voxel_dim # + 0.5/float(voxel_dim)
                    expand_point.append(grid_point)
                    # expand_label.append(label)
                point_cloud = point_cloud - 0.5
                expand_point = torch.stack(expand_point)
                expand_point = expand_point - 0.5

                expand_label = torch.zeros_like(expand_point)[:,:,0]
                # expand_label = torch.stack(expand_label)
                start_time = time.time()
                net_out_all, acc, acc2 = self.bsp_network(expand_label, expand_point, point_cloud, point_voxel, torch.tensor([1.0]), is_training=False)
                total_time += (time.time()-start_time)
                point_cloud = point_cloud*padding
                # clean_point = clean_point*padding
                # shift_pc = shift_pc*padding
                acc_mean += acc
                acc_list.append(acc)
                # detect_point_list = torch.chunk(expand_point, 200000, dim=1)
                # pred_point_list = torch.chunk(pred, 200000, dim=0)
                # net_out_all_list = []
                # for idx, d in enumerate(detect_point_list):
                #     pred_pc = d[0][torch.where(pred_point_list[idx] > 0.5)].cpu().numpy()
                #     net_out_all_list.append(pred_pc)
                # output = np.concatenate(net_out_all_list, 0)
                #
                # np.savetxt(config.sample_dir + "/" + self.txt_data[t] + "_predpc.txt",
                #            output, delimiter=";")

                # total_num += 1
                # # point_cloud = [x / scale[idx] for idx, x in enumerate(point_cloud)]
                mesh = extract_mesh_tsdf(net_out_all, padding=padding)
                # # trimesh.smoothing.filter_laplacian(mesh, iterations=1)
                mesh.export(config.sample_dir + "/" + self.txt_data[t] + ".off")
                np.savetxt(config.sample_dir + "/" + self.txt_data[t] + "_pc.txt",
                           (point_cloud[0]).cpu().detach().numpy(), delimiter=";")
                # # np.savetxt(config.sample_dir + "/" + self.txt_data[t] + "_shiftpc.txt",
                # #            (shift_pc[0]).cpu().detach().numpy(), delimiter=";")
                # # np.savetxt(config.sample_dir + "/" + self.txt_data[t] + "_cleanpc.txt",
                # #            (clean_point[0]).cpu().detach().numpy(), delimiter=";")

                print("[sample%d]"%t, acc)
                # if t > 8:
                #     break
        # print("inference time :", total_time/10)
        index = torch.topk(torch.tensor(acc_list), k=min(10, data_num), largest=False)[1]
        # for idx in index:
        #     print(self.txt_data[idx])
        print("box_predict acc is： ", acc_mean/total_num)



    def test_iou(self, config):
        # load previous checkpoint
        result_path = config.sample_dir + "/" + self.txt_data[0][:8]
        if not os.path.exists(result_path):
            os.mkdir(result_path)
        if not self.load(): exit(-1)
        self.bsp_network.train()
        acc_mean = 0
        acc2_mean = 0
        total_num = 0
        acc_list = []
        data_num = min(len(self.txt_data), config.end) - config.start
        padding=1.1

        avg_loss_sp = 0
        avg_loss_chamfer = 0
        avg_loss_tt = 0
        end = min(len(self.txt_data), config.end)
        with torch.no_grad():
            for t in range(config.start, end):
                filename = self.txt_data[t:t + 1]
                point_cloud, query_p, query_l, clean_point = load_data_test_iou(self.txt_data, [t], self.root_data_dir)
                # point_cloud, query_p, query_l, clean_point, points_chamfer = load_data(self.txt_data, [t],
                #                                                                        self.root_data_dir)

                query_p = query_p.to(self.device)/padding
                query_l = query_l.to(self.device)
                point_cloud = point_cloud.to(self.device)/padding
                clean_point = clean_point.to(self.device)/padding
                # point_cloud = clean_point
                self.bsp_network.zero_grad()
                loss, acc = self.loss(query_l, query_p, point_cloud,
                                      self.bsp_network.local_decoder, point_cloud)
                # np.savetxt(config.sample_dir + "/" + self.txt_data[t] + "_wrong.txt", wrong_point.cpu().numpy(), delimiter=";")
                # avg_loss_tt += acc
                print(str(
                    self.sample_vox_size) + " Epoch: [%2d/%2d] time: %4.4f, iou: %.6f" % (
                          t, end, 0.0, acc))
                acc_list.append(acc)
                # if t == 2:
                #     break
        index = torch.topk(torch.tensor(acc_list), k=min(10, data_num), largest=False)[1]
        for idx in index:
            print(idx, self.txt_data[idx], acc_list[idx])
        print("mean iou is： ", np.array(acc_list).mean())


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


def extract_mesh_tsdf(voxel, padding=1.0):
    if isinstance(voxel, numpy.ndarray):
        voxel = torch.from_numpy(voxel)
    voxel = voxel.squeeze()
    n_x, n_y, n_z = voxel.shape
    voxel = F.pad(voxel, (1, 1, 1, 1, 1, 1), 'constant', -100)
    voxel = voxel.cpu().detach().numpy()

    vertices, triangles = mcubes.marching_cubes(voxel, math.log(0.2))
    vertices -= 1
    vertices /= np.array([n_x, n_y, n_z])
    vertices -= 0.5
    vertices *= padding
    # # Undo padding
    # matrix = rotation(-math.pi / 2).numpy()
    # vertices = np.matmul(vertices, matrix)
    mesh = trimesh.Trimesh(vertices, triangles, process=False)
    return mesh


def extract_mesh_voxel(voxel):
    if isinstance(voxel, numpy.ndarray):
        voxel = torch.from_numpy(voxel)
    voxel = voxel.squeeze()
    n_x, n_y, n_z = voxel.shape
    voxel = F.pad(voxel, (1, 1, 1, 1, 1, 1), 'constant', 0)
    voxel = voxel.cpu().detach().numpy()

    vertices, triangles = mcubes.marching_cubes(voxel, 0.5)
    vertices -= 1
    vertices /= np.array([n_x, n_y, n_z])
    vertices -= 0.5
    # # Undo padding
    # matrix = rotation(-math.pi / 2).numpy()
    # vertices = np.matmul(vertices, matrix)
    mesh = trimesh.Trimesh(vertices, triangles, process=False)
    return mesh
