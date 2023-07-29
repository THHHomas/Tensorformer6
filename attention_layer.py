import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from utils import index_points, knn_point, farthest_point_sample
from transformer import TransformerBlock, CTransformerBlock, TLinear, vector_atnettion, dt_scaled_atnettion
from transformer_pos_enc import get_3d_sincos_pos_embed_from_point
import matplotlib.pyplot as plt


def generate_grad(weight):
    weight_norm_soft = F.softmax(weight, dim=0).cpu().detach().numpy()
    weight = weight.cpu().detach().numpy()

    grad_soft = (1 - weight_norm_soft) * weight_norm_soft

    weight_sum = np.abs(weight).sum()
    weight_norm = np.abs(weight) / weight_sum
    grad = (1 - weight_norm) / weight_sum
    return grad, grad_soft


def viaualize_softmax_grad(weight):
    grad1, grad_soft1 = generate_grad(weight=weight[0, 0, 0])
    grad2, grad_soft2 = generate_grad(weight=weight[0, 0, 1])
    fsize = 36
    x = np.arange(weight.shape[-1])
    plt.subplot(1,2,1)
    plt.ylabel('gradient', fontsize=fsize)
    plt.ylim(0, 0.5)
    plt.xticks(size=fsize)
    plt.yticks(size=fsize)
    plt.plot(x, grad_soft1, color="blue")
    plt.plot(x, grad1, color="red")
    plt.subplot(1, 2, 2)
    plt.plot(x, grad_soft2, color="blue", label='softmax gradient')
    plt.plot(x, grad2, color="red", label='ours')
    plt.xticks(size=fsize)
    plt.yticks(size=fsize)
    plt.legend(loc='upper right', fontsize=fsize)
    plt.xlabel('feature dimension', fontsize=fsize)

    plt.ylim(0,0.5)
    plt.show()


def visualize_grad(weight):
    weight = weight[0, 0, 0]
    weight = weight.cpu().detach().numpy()
    weight_sum = np.abs(weight).sum()
    weight_norm = np.abs(weight) /weight_sum
    grad = (1-weight_norm)/weight_sum
    x = np.arange(weight.shape[0])
    plt.plot(x, grad)
    plt.ylim(0, 1)
    plt.show()


class transformer_layer2(nn.Module):
    def __init__(self, input_dim, out_dim):
        super(transformer_layer2, self).__init__()
        self.out_dim = out_dim
        self.prefix_linear = nn.Sequential(nn.Linear(input_dim, out_dim),
                                           nn.ReLU(),
                                           nn.Linear(out_dim, out_dim))
        self.linear_r = nn.Linear(out_dim, out_dim)
        self.linear_c = nn.Linear(out_dim, out_dim)
        self.linear_cc = nn.Linear(out_dim, out_dim)
        self.linear_ccc = nn.Linear(out_dim, out_dim)
        self.linear_cccc = nn.Linear(out_dim, out_dim)
        self.linear_ccccc = nn.Linear(out_dim, out_dim)
        self.linear_cccccc = nn.Linear(out_dim, out_dim)
        # self.linear_t = nn.Linear(input_dim, out_dim)
        self.fc_position = nn.Sequential(
            nn.Linear(3, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim)
        )

        self.suffix_linear = nn.Linear(out_dim, out_dim)
        self.suffix_bn = nn.BatchNorm1d(out_dim)
        # self.Matrix = nn.Parameter(torch.Tensor(out_dim, 1, 1, out_dim))
        # self.Bais = nn.Parameter(torch.Tensor(out_dim))
        # nn.init.xavier_uniform_(self.Matrix)
        # nn.init.constant_(self.Bais, 0)

    def forward(self, feature, xyz, knn_num=36, gl=False):
        # feature: BNC
        B, N, _ = feature.shape
        if gl:
            idx = [torch.from_numpy(np.random.choice(N, knn_num)) for _ in range(B)]
            point_index = torch.stack(idx).unsqueeze(1).repeat(1, N, 1)
        else:
            point_index = knn_point(knn_num, xyz, xyz)

        # relative_xyz = index_points(xyz, point_index)
        # relative_xyz = relative_xyz - xyz.unsqueeze(2)
        # position_weight = self.fc_position(relative_xyz)

        feature = F.relu(self.prefix_linear(feature))
        r, s, t = feature, feature, feature
        weight = self.linear_r(index_points(r, point_index) - s.unsqueeze(2)).reshape(B, N, self.out_dim, 1, -1)
        group_feature = index_points(t, point_index).reshape(B, N, self.out_dim, -1, 1)
        weight = F.softmax(weight/math.sqrt(weight.shape[-3]), dim=-1)
        feature = torch.matmul(weight, group_feature).squeeze()  # / math.sqrt(weight.shape[-1])

        # # merge in k_nn dim
        # weight = self.linear_r(index_points(r, point_index) - s.unsqueeze(2))
        # group_feature = index_points(t, point_index)
        # feature = torch.matmul(group_feature.permute(0,1,3,2).unsqueeze(3),
        #                        weight.permute(0,1,3,2).unsqueeze(4)).squeeze()/math.sqrt(knn_num)
        # # merge in channel dim
        # feature = self.linear_c(feature)
        #

        if B == 1:
            feature = feature.unsqueeze(0)
        feature = self.suffix_linear(feature)
        return feature, N


class pointconv(nn.Module):
    def __init__(self, input_dim, out_dim, knn=24):
        super(pointconv, self).__init__()
        self.out_dim = out_dim
        self.prefix_linear = nn.Sequential(nn.Linear(input_dim, out_dim),
                                           nn.ReLU(),
                                           nn.Linear(out_dim, out_dim))
        self.linear_r = nn.Linear(3, out_dim * out_dim)

        self.linear_q = nn.Linear(out_dim, out_dim)
        self.linear_k = nn.Linear(out_dim, out_dim)
        self.linear_v = nn.Linear(out_dim, out_dim)
        self.layer_norm = nn.LayerNorm(knn * out_dim)
        self.knn = knn
        self.fc_position = nn.Sequential(
            nn.Linear(3, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim)
        )

        self.suffix_linear = nn.Linear(out_dim, out_dim)
        self.suffix_bn = nn.BatchNorm1d(out_dim)
        # self.Matrix = nn.Parameter(torch.Tensor(out_dim, 1, 1, out_dim))
        self.std = nn.Parameter(torch.Tensor(out_dim))
        # nn.init.xavier_uniform_(self.Matrix)
        nn.init.constant_(self.std, 1)

    def forward(self, feature, xyz, knn_num=24, gl=False):
        # feature: BNC
        B, N, _ = feature.shape
        if gl:
            idx = [torch.from_numpy(np.random.choice(N, knn_num)) for _ in range(B)]
            point_index = torch.stack(idx).unsqueeze(1).repeat(1, N, 1)
        else:
            point_index = knn_point(knn_num, xyz, xyz)

        relative_xyz = index_points(xyz, point_index)
        relative_xyz = relative_xyz - xyz.unsqueeze(2)
        # pos_enc = self.fc_position(relative_xyz)

        feature = F.relu(self.prefix_linear(feature))
        # q, k, v = F.relu(self.linear_q(feature)), F.relu(self.linear_k(feature)), F.relu(self.linear_v(feature))
        q, k, v = feature, feature, feature
        weight = self.linear_r(relative_xyz).reshape(B, N, -1, self.out_dim)
        # weight = weight / (weight.abs() + 1e-5).sum(-2, keepdim=True) * math.sqrt(self.knn)
        weight = F.softmax(weight, dim=-2)
        # weight = weight / math.sqrt(knn_num*self.out_dim)
        # weight = weight.permute(0,1,3,2)
        # weight = self.layer_norm(weight) + 1.0/(knn_num*self.out_dim)
        # weight = weight.permute(0, 1, 3, 2)
        # weight = (weight - weight.mean(dim=-2, keepdim=True))\
        #          *(F.relu(self.std)+1e-1)/np.sqrt(self.out_dim) + 1.0/(knn_num*self.out_dim)
        # weight.std(dim=)
        # ss = weight.sum(-2)
        group_feature = index_points(v, point_index)
        feature = torch.matmul(group_feature.reshape(B, N, 1, -1),
                               weight).squeeze()  # /math.sqrt(knn_num*self.out_dim)
        # feature = self.layer_norm(feature)
        if B == 1:
            feature = feature.unsqueeze(0)

        feature = self.suffix_linear(feature)
        return feature, N


class transformer_layer(nn.Module):
    def __init__(self, input_dim, out_dim):
        super(transformer_layer, self).__init__()
        self.out_dim = out_dim
        self.input_dim = input_dim
        self.prefix_linear = nn.Sequential(nn.Linear(input_dim, out_dim),
                                           nn.ReLU(),
                                           nn.Linear(out_dim, out_dim))
        self.linear_r = nn.Linear((input_dim+3)*2, out_dim * out_dim)

        self.linear_q = nn.Linear(out_dim, out_dim)
        self.linear_k = nn.Linear(out_dim, out_dim)
        self.linear_v = nn.Linear(input_dim+3, out_dim)

        self.fc_position = nn.Sequential(
            nn.Linear(3, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim)
        )

        self.suffix_linear = nn.Linear(out_dim, out_dim)
        self.suffix_bn = nn.BatchNorm1d(out_dim)
        # self.Matrix = nn.Parameter(torch.Tensor(out_dim, 1, 1, out_dim))
        self.std = nn.Parameter(torch.Tensor(out_dim))
        # nn.init.xavier_uniform_(self.Matrix)
        nn.init.constant_(self.std, 1)

    def forward(self, feature, xyz, knn_num=36, gl=False):
        # feature: BNC
        B, N, _ = feature.shape
        point_index = knn_point(knn_num, xyz, xyz)

        # relative_xyz = index_points(xyz, point_index)
        # relative_xyz = relative_xyz - xyz.unsqueeze(2)
        # pos_enc = self.fc_position(relative_xyz)

        # feature = F.relu(self.prefix_linear(feature))
        # q, k, v = F.relu(self.linear_q(feature)), F.relu(self.linear_k(feature)), F.relu(self.linear_v(feature))
        q, k, v = feature, feature, feature
        pre_weight = torch.cat([q, xyz], -1)
        g_weight = index_points(pre_weight, point_index) - pre_weight.unsqueeze(2)
        g_weight = torch.cat([g_weight, pre_weight.unsqueeze(2).repeat(1,1,knn_num,1)], -1)

        weight = self.linear_r(g_weight).reshape(B, N, -1, self.out_dim)
        # visualize_grad(weight)
        # if self.out_dim==32:
        #     viaualize_softmax_grad(weight)

        weight_abs = weight.abs() + 1e-7
        weight =  weight / weight_abs.sum(-1, keepdim=True)*math.sqrt(self.out_dim)
        # weight = F.softmax(weight, dim=-1)
        # weight = weight / math.sqrt(knn_num*self.out_dim)
        # weight = weight.permute(0,1,3,2)
        # weight = self.layer_norm(weight) + 1.0/(knn_num*self.out_dim)
        # weight = weight.permute(0, 1, 3, 2)
        # weight = (weight - weight.mean(dim=-2, keepdim=True))\
        #          *(F.relu(self.std)+1e-1)/np.sqrt(self.out_dim) + 1.0/(knn_num*self.out_dim)
        # weight.std(dim=)
        # ss = weight.sum(-2)
        group_feature = index_points(F.relu(self.linear_v(torch.cat([v, xyz], -1))), point_index)
        feature = torch.matmul(group_feature.reshape(B, N, 1, -1),
                               weight).squeeze()  # /math.sqrt(knn_num*self.out_dim)
        # feature = self.layer_norm(feature)
        if B == 1:
            feature = feature.unsqueeze(0)

        # feature = self.suffix_bn(feature.permute(0,2,1)).permute(0,2,1)
        feature = self.suffix_linear(feature)
        return feature, N


class transformer_layer_sub(nn.Module):
    def __init__(self, input_dim, out_dim):
        super(transformer_layer_sub, self).__init__()
        self.out_dim = out_dim
        # self.beta_num = 24
        self.input_dim = input_dim
        self.knn_num = 36
        self.prefix_linear = nn.Sequential(nn.Linear(input_dim, out_dim),
                                           nn.ReLU(),
                                           nn.Linear(out_dim, out_dim))
        self.linear_r = nn.Linear(out_dim*self.knn_num, self.knn_num*out_dim)
        # self.linear_g = nn.Linear(out_dim*self.knn_num, self.beta_num*out_dim)

        self.linear_q = nn.Linear(input_dim, out_dim)
        self.linear_k = nn.Linear(out_dim, out_dim)
        self.linear_v = nn.Linear(input_dim, out_dim)
        self.embed_dim = 60
        self.fc_delta = nn.Sequential(
            nn.Linear(self.embed_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim))

        self.suffix_linear = nn.Linear(out_dim, out_dim)
        self.suffix_bn = nn.BatchNorm1d(out_dim)
        # self.Matrix = nn.Parameter(torch.Tensor(out_dim, 1, 1, out_dim))
        self.std = nn.Parameter(torch.Tensor(out_dim))
        # nn.init.xavier_uniform_(self.Matrix)
        nn.init.constant_(self.std, 1)

    def forward(self, feature, xyz, gl=False):
        # feature: BNC
        B, N, _ = feature.shape
        point_index = knn_point(self.knn_num, xyz, xyz)

        # relative_xyz = index_points(xyz, point_index)
        # relative_xyz = relative_xyz - xyz.unsqueeze(2)
        # pos_enc = self.fc_position(relative_xyz)

        # feature = F.relu(self.prefix_linear(feature))
        # q, k, v = F.relu(self.linear_q(feature)), F.relu(self.linear_k(feature)), F.relu(self.linear_v(feature))
        q, k, v = F.relu(self.linear_q(feature)), feature, F.relu(self.linear_v(feature))
        # pre_weight = torch.cat([q, xyz], -1)
        position = index_points(xyz, point_index)-xyz.unsqueeze(2)

        position_enc = get_3d_sincos_pos_embed_from_point(self.embed_dim, position)
        position_enc = self.fc_delta(position_enc)

        g_weight = index_points(q, point_index) - q.unsqueeze(2)
        g_weight = (g_weight + position_enc).reshape(B, N, -1)
        weight = self.linear_r(g_weight).reshape(B, N, -1, self.out_dim)

        weight_abs = weight.abs() + 1e-7
        weight = weight / weight_abs.sum(-1, keepdim=True) * math.sqrt(self.out_dim)
        # weight = F.softmax(weight, dim=-1)
        # weight = weight / math.sqrt(knn_num*self.out_dim)
        # weight = weight.permute(0,1,3,2)
        # weight = self.layer_norm(weight) + 1.0/(knn_num*self.out_dim)
        # weight = weight.permute(0, 1, 3, 2)
        # weight = (weight - weight.mean(dim=-2, keepdim=True))\
        #          *(F.relu(self.std)+1e-1)/np.sqrt(self.out_dim) + 1.0/(knn_num*self.out_dim)
        # weight.std(dim=)
        # ss = weight.sum(-2)
        group_feature = index_points(v, point_index) + position_enc
        # group_feature = self.linear_g(group_feature.reshape(B, N, -1)).reshape(B, N, -1, self.out_dim)
        # feature = torch.matmul(weight.unsqueeze(2), group_feature).squeeze()  # /math.sqrt(knn_num*self.out_dim)
        feature = torch.matmul(group_feature.permute(0,1,3,2).unsqueeze(3), weight.permute(0,1,3,2).unsqueeze(4)).squeeze()
        # feature = self.layer_norm(feature)
        if B == 1:
            feature = feature.unsqueeze(0)

        # feature = self.suffix_bn(feature.permute(0,2,1)).permute(0,2,1)
        feature = self.suffix_linear(feature)
        return feature, N


class indicator_layer(nn.Module):
    def __init__(self, input_dim, out_dim):
        super(indicator_layer, self).__init__()
        self.prefix_linear = nn.Linear(input_dim, out_dim)

        self.fc_position = nn.Sequential(
            nn.Linear(3, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim)
        )
        self.fc_position2 = nn.Sequential(
            nn.Linear(3, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim * 2)
        )
        self.fc_matrix = nn.Linear(input_dim, 12)
        self.linear_merge = nn.Linear(out_dim * 2, out_dim)
        self.linear_r = nn.Linear(out_dim, 1)
        self.linear_t = nn.Linear(out_dim, out_dim)

        self.suffix_linear = nn.Linear(out_dim, out_dim)
        self.out_dim = out_dim
        self.fc = nn.Linear(out_dim * 2, out_dim)

    def forward(self, feature1, xyz1, xyz2, normal=None, fps_num=None, knn_num=16):
        B, N, _ = xyz1.shape
        feature1 = self.prefix_linear(feature1)

        point_index, distance = knn_point(knn_num, xyz1, xyz2, dis=True)
        # 0.
        min_distance = distance[:,:,0]
        weight = torch.ones_like(min_distance)
        weight[torch.where(min_distance > 0.03)] = 10
        # min_distance = distance[:,:,0:4].mean(-1)
        # r4 = torch.clamp(4*min_distance, max=0.4, min=0.02)
        # weight = torch.zeros_like(distance)
        # weight[torch.where(distance < r4.unsqueeze(2))] = 1.0
        # ss = (torch.sum(weight, -1, keepdim=True)+0.1)
        # weight = (weight/ss*knn_num).unsqueeze(3).unsqueeze(2)
        # distance = torch.softmax(distance*200, -1).unsqueeze(3).unsqueeze(2)

        g_feature = index_points(feature1, point_index)

        g_xyz = index_points(xyz1, point_index)
        g_xyz = g_xyz - xyz2.unsqueeze(2)

        position_weight = (self.fc_position(g_xyz))
        # position_weight = torch.softmax(position_weight, -2)
        new_feature = torch.matmul(position_weight.permute(0, 1, 3, 2).unsqueeze(3),
                               g_feature.permute(0, 1, 3, 2).unsqueeze(4)).squeeze() / math.sqrt(knn_num)

        # g_feature = torch.cat([feature.unsqueeze(1), g_feature], 1)
        # attention_weight = self.linear_t(g_feature-feature.unsqueeze(1))
        # feature = torch.matmul(attention_weight.permute(0,2,1).unsqueeze(2), g_feature.permute(0,2,1).unsqueeze(3)).squeeze()/math.sqrt(knn_num)

        # new_feature.append(feature)
        if len(new_feature.shape) == 2:
            new_feature = new_feature.unsqueeze(0)
        return new_feature, weight


class indicator_layer2(nn.Module):
    def __init__(self, input_dim, out_dim):
        super(indicator_layer2, self).__init__()
        self.prefix_linear = nn.Linear(input_dim, out_dim)
        self.fc_position = nn.Sequential(
            nn.Linear(3, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim)
        )
        self.fc_position2 = nn.Sequential(
            nn.Linear(3, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim * 2)
        )
        self.fc_matrix = nn.Linear(input_dim, 12)
        self.linear_merge = nn.Linear(out_dim * 2, out_dim)
        self.linear_r = nn.Linear(out_dim, 1)
        self.linear_t = nn.Linear(out_dim, out_dim)

        self.suffix_linear = nn.Linear(out_dim, out_dim)
        self.out_dim = out_dim
        self.fc = nn.Linear(out_dim * 2, out_dim)

    def forward(self, feature1, xyz1, xyz2, normal=None, fps_num=None, knn_num=8):
        # feature: BNC
        # B, N, _ = xyz1.shape
        # new_feature = []
        # for idx in range(len(xyz2)):
        #     xyz_2 = xyz2[idx].unsqueeze(0)
        #     xyz_1 = xyz1[idx].unsqueeze(0)
        #     feature_1 = feature1[idx].unsqueeze(0)
        #     feature_1 = self.prefix_linear(feature_1)
        #     point_index = knn_point(knn_num, xyz_1, xyz_2)
        #     g_feature = index_points(feature_1, point_index).squeeze()
        #     g_xyz = index_points(xyz_1, point_index)
        #     g_xyz = g_xyz - xyz_2.unsqueeze(2)
        #     position_weight = self.fc_position(g_xyz).squeeze()
        #     feature = torch.matmul(position_weight.permute(0, 2, 1).unsqueeze(2),
        #                            g_feature.permute(0, 2, 1).unsqueeze(3)).squeeze() / math.sqrt(knn_num)
        #     new_feature.append(feature)
        #
        B, N, _ = xyz1.shape
        new_feature = []
        for idx in range(len(xyz2)):
            xyz_2 = xyz2[idx].unsqueeze(0)
            xyz_1 = xyz1[idx].unsqueeze(0)
            feature_1 = feature1[idx].unsqueeze(0)
            feature_1 = self.prefix_linear(feature_1)

            point_index = knn_point(1, xyz_1, xyz_2)

            g_feature = index_points(feature_1, point_index).squeeze()

            # g_feature = torch.cat([feature.unsqueeze(1), g_feature], 1)
            # attention_weight = self.linear_t(g_feature-feature.unsqueeze(1))
            # feature = torch.matmul(attention_weight.permute(0,2,1).unsqueeze(2), g_feature.permute(0,2,1).unsqueeze(3)).squeeze()/math.sqrt(knn_num)

            new_feature.append(g_feature)
        return new_feature


class local_decoder(nn.Module):
    def __init__(self, ef_dim):
        super(local_decoder, self).__init__()
        # ef_dim = 32
        self.ef_dim2 = ef_dim
        self.ef_dim = ef_dim
        transformer_layer = transformer_layer_sub
        self.tl1 = transformer_layer(3+0, self.ef_dim // 4)
        self.bn_tl1 = nn.BatchNorm1d(self.ef_dim // 4)
        self.tl2 = transformer_layer(self.ef_dim // 4, self.ef_dim)
        self.bn_tl2 = nn.BatchNorm1d(self.ef_dim)
        self.res_fc = nn.Linear(self.ef_dim, self.ef_dim)
        self.tl3 = transformer_layer(self.ef_dim, self.ef_dim)
        self.bn_tl3 = nn.BatchNorm1d(self.ef_dim)
        self.tl4 = transformer_layer(self.ef_dim, self.ef_dim)
        self.bn_tl4 = nn.BatchNorm1d(self.ef_dim)
        # self.tl5 = transformer_layer(self.ef_dim, self.ef_dim)
        # self.bn_tl5 = nn.BatchNorm1d(self.ef_dim)
        # self.tl6 = transformer_layer(self.ef_dim, self.ef_dim)
        # self.bn_tl6 = nn.BatchNorm1d(self.ef_dim)

        self.indicator3 = indicator_layer(self.ef_dim, self.ef_dim2 *4)
        # self.indicator4 = indicator_layer(self.ef_dim, self.ef_dim2 * 2)

        self.cls = nn.Sequential(nn.Linear(self.ef_dim2 * 4, self.ef_dim2 // 4),
                                 nn.ReLU(),
                                 nn.Linear(self.ef_dim2 // 4, 2))
        self.orientation = nn.Sequential(nn.Linear(self.ef_dim, self.ef_dim2 // 4),
                                 nn.ReLU(),
                                 nn.Linear(self.ef_dim2 // 4, 3),
                                nn.Tanh())

        # self.sdf = nn.Sequential(nn.Linear(self.ef_dim2 * 4, self.ef_dim2 // 2),
        #                          nn.ReLU(),
        #                          nn.Linear(self.ef_dim2 // 2, 1),
        #                          nn.Tanh())

    def forward(self, xyz, detect_point, normal_gt):
        B = xyz.shape[0]
        # init_feature = get_3d_sincos_pos_embed_from_point(12, xyz)
        # init_feature = torch.cat([init_feature, xyz], -1)
        feature1 = self.tl1(xyz, xyz)[0]
        feature1 = self.bn_tl1(feature1.permute(0, 2, 1)).permute(0, 2, 1)
        feature2 = self.tl2(feature1, xyz)[0]
        feature2 = self.bn_tl2(feature2.permute(0, 2, 1)).permute(0, 2, 1)

        feature3 = self.tl3(feature2, xyz)[0]
        feature3 = self.bn_tl3(feature3.permute(0, 2, 1)).permute(0, 2, 1)
        feature4 = self.tl4(feature3, xyz)[0]
        feature4 = self.bn_tl4(feature4.permute(0, 2, 1)).permute(0, 2, 1)
        feature4 += self.res_fc(feature2)
        # feature5 = self.tl5(feature4, xyz)[0]
        # feature5 = self.bn_tl5(feature5.permute(0, 2, 1)).permute(0, 2, 1)
        # feature6 = self.tl6(feature5, xyz)[0]
        # feature6 = self.bn_tl6(feature6.permute(0, 2, 1)).permute(0, 2, 1)
        # feature6 += feature4
        # ori = self.orientation(feature4)
        # ori = ori/(torch.norm(ori, p=2, dim=-1, keepdim=True) + 1e-6)

        new_feature3, weight = self.indicator3(feature4, xyz, detect_point, fps_num=0, knn_num=12)
        # index = farthest_point_sample(xyz, 127)
        # global_feature = index_points(feature4, index)
        # global_xyz = index_points(xyz, index)
        # new_feature4 = self.indicator4(global_feature, global_xyz, detect_point, normal=normal_gt, fps_num=0, knn_num=127)

        # detect_point_list = torch.chunk(detect_point[0], 10)
        # new_feature3_list = []
        # for d in detect_point_list:
        #     new_feature3 = self.indicator3(feature4, xyz, [d], normal=normal_gt, fps_num=0, knn_num=12)
        #     new_feature3_list += new_feature3
        # new_feature3 = [torch.cat(new_feature3_list, 0)]

        occ = []
        sdf_list = []
        for i in range(B):
            # new_feature = torch.cat([new_feature4[i], new_feature3[i]], -1)
            new_feature = new_feature3[i]
            logit = self.cls(new_feature)
            occ.append(logit)
            # sdf = self.sdf(new_feature)*0.06
            # sdf_list.append(sdf)
        return occ, weight


class seg_decoder(nn.Module):
    def __init__(self, ef_dim):
        super(seg_decoder, self).__init__()
        # ef_dim = 32
        self.ef_dim2 = ef_dim
        self.ef_dim = ef_dim
        # transformer_layer = TransformerBlock
        self.tl1 = transformer_layer(3, self.ef_dim // 4)
        self.bn_tl1 = nn.BatchNorm1d(self.ef_dim // 4)
        self.tl2 = transformer_layer(self.ef_dim // 4, self.ef_dim)
        self.bn_tl2 = nn.BatchNorm1d(self.ef_dim)
        self.res_fc = nn.Linear(self.ef_dim, self.ef_dim)
        self.tl3 = transformer_layer(self.ef_dim, self.ef_dim)
        self.bn_tl3 = nn.BatchNorm1d(self.ef_dim)


        self.tl4 = transformer_layer(self.ef_dim, self.ef_dim)
        self.bn_tl4 = nn.BatchNorm1d(self.ef_dim)

        self.tl5 = transformer_layer(self.ef_dim, self.ef_dim)
        self.bn_tl5 = nn.BatchNorm1d(self.ef_dim)
        self.interpolate5 = indicator_layer(self.ef_dim, self.ef_dim)

        self.tl4_back = transformer_layer(self.ef_dim, self.ef_dim)
        self.bn_tl4_back = nn.BatchNorm1d(self.ef_dim)
        self.interpolate4 = indicator_layer(self.ef_dim, self.ef_dim)

        self.tl3_back = transformer_layer(self.ef_dim, self.ef_dim)
        self.bn_tl3_back = nn.BatchNorm1d(self.ef_dim)

        self.indicator3 = indicator_layer(self.ef_dim*2, self.ef_dim2 *4)
        # self.indicator4 = indicator_layer(self.ef_dim, self.ef_dim2 * 2)

        self.cls = nn.Sequential(nn.Linear(self.ef_dim2 * 4, self.ef_dim2 // 4),
                                 nn.ReLU(),
                                 nn.Linear(self.ef_dim2 // 4, 2))
        self.orientation = nn.Sequential(nn.Linear(self.ef_dim, self.ef_dim2 // 4),
                                 nn.ReLU(),
                                 nn.Linear(self.ef_dim2 // 4, 3),
                                nn.Tanh())

        # self.sdf = nn.Sequential(nn.Linear(self.ef_dim2 * 4, self.ef_dim2 // 2),
        #                          nn.ReLU(),
        #                          nn.Linear(self.ef_dim2 // 2, 1),
        #                          nn.Tanh())

    def forward(self, xyz, detect_point, normal_gt):
        B = xyz.shape[0]
        feature1 = self.tl1(xyz, xyz)[0]
        feature1 = self.bn_tl1(feature1.permute(0, 2, 1)).permute(0, 2, 1)
        feature2 = self.tl2(feature1, xyz)[0]
        feature2 = self.bn_tl2(feature2.permute(0, 2, 1)).permute(0, 2, 1)

        feature3 = self.tl3(feature2, xyz)[0]
        feature3 = self.bn_tl3(feature3.permute(0, 2, 1)).permute(0, 2, 1)

        # down 512
        far_idx512 = farthest_point_sample(xyz, 512)
        feature3_down512 = index_points(feature3, far_idx512)
        xyz512 = index_points(xyz, far_idx512)
        feature4 = self.tl4(feature3_down512, xyz512)[0]
        feature4 = self.bn_tl4(feature4.permute(0, 2, 1)).permute(0, 2, 1)

        # up 3000
        feature4_up3000, _ = self.interpolate5(feature4, xyz512, xyz, fps_num=0, knn_num=12)
        # feature4_up3000 += feature3
        feature5 = self.tl5(feature4_up3000, xyz)[0]
        feature5 = self.bn_tl5(feature5.permute(0, 2, 1)).permute(0, 2, 1)

        feature = torch.cat([feature3, feature5], -1)

        new_feature3, weight = self.indicator3(feature, xyz, detect_point, fps_num=0, knn_num=12)
        occ = []
        sdf_list = []
        for i in range(B):
            # new_feature = torch.cat([new_feature4[i], new_feature3[i]], -1)
            new_feature = new_feature3[i]
            logit = self.cls(new_feature)
            occ.append(logit)
            # sdf = self.sdf(new_feature)*0.06
            # sdf_list.append(sdf)
        return occ, weight