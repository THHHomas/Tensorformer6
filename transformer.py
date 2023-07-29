from utils import index_points, square_distance
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from utils import knn_point
from transformer_pos_enc import get_1d_sincos_pos_embed_from_grid



class vector_atnettion(nn.Module):
    def __init__(self, input_dim, out_dim):
        super(vector_atnettion, self).__init__()
        self.out_dim = out_dim
        self.input_dim = input_dim
        self.prefix_linear = nn.Sequential(nn.Linear(input_dim, out_dim),
                                           nn.ReLU(),
                                           nn.Linear(out_dim, out_dim))
        self.linear_r = nn.Linear((input_dim+3)*2, out_dim)

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

        # weight_abs = weight.abs() + 1e-7
        # weight =  weight / weight_abs.sum(-1, keepdim=True)*math.sqrt(self.out_dim)
        weight = F.softmax(weight, dim=-1)
        # weight = weight / math.sqrt(knn_num*self.out_dim)
        # weight = weight.permute(0,1,3,2)
        # weight = self.layer_norm(weight) + 1.0/(knn_num*self.out_dim)
        # weight = weight.permute(0, 1, 3, 2)
        # weight = (weight - weight.mean(dim=-2, keepdim=True))\
        #          *(F.relu(self.std)+1e-1)/np.sqrt(self.out_dim) + 1.0/(knn_num*self.out_dim)
        # weight.std(dim=)
        # ss = weight.sum(-2)
        group_feature = index_points(F.relu(self.linear_v(torch.cat([v, xyz], -1))), point_index)
        feature = torch.matmul(group_feature.permute(0,1,3,2).unsqueeze(3),
                               weight.permute(0,1,3,2).unsqueeze(4)).squeeze()  # /math.sqrt(knn_num*self.out_dim)
        # feature = self.layer_norm(feature)
        if B == 1:
            feature = feature.unsqueeze(0)

        # feature = self.suffix_bn(feature.permute(0,2,1)).permute(0,2,1)
        feature = self.suffix_linear(feature)
        return feature, N


class dt_scaled_atnettion(nn.Module):
    def __init__(self, input_dim, out_dim):
        super(dt_scaled_atnettion, self).__init__()
        self.out_dim = out_dim
        self.input_dim = input_dim
        self.prefix_linear = nn.Sequential(nn.Linear(input_dim, out_dim),
                                           nn.ReLU(),
                                           nn.Linear(out_dim, out_dim))
        self.linear_r = nn.Linear((input_dim+3)*2, out_dim)

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
        # pre_weight = torch.cat([q, xyz], -1)
        g_f = index_points(q, point_index)
        weight = torch.matmul(g_f, q.unsqueeze(3))

        # visualize_grad(weight)
        # if self.out_dim==32:
        #     viaualize_softmax_grad(weight)

        # weight_abs = weight.abs() + 1e-7
        # weight =  weight / weight_abs.sum(-1, keepdim=True)*math.sqrt(self.out_dim)
        weight = F.softmax(weight, dim=-2)
        # weight = weight / math.sqrt(knn_num*self.out_dim)
        # weight = weight.permute(0,1,3,2)
        # weight = self.layer_norm(weight) + 1.0/(knn_num*self.out_dim)
        # weight = weight.permute(0, 1, 3, 2)
        # weight = (weight - weight.mean(dim=-2, keepdim=True))\
        #          *(F.relu(self.std)+1e-1)/np.sqrt(self.out_dim) + 1.0/(knn_num*self.out_dim)
        # weight.std(dim=)
        # ss = weight.sum(-2)
        group_feature = index_points(F.relu(self.linear_v(torch.cat([v, xyz], -1))), point_index)
        feature = torch.matmul(group_feature.permute(0,1,3,2),
                               weight).squeeze()  # /math.sqrt(knn_num*self.out_dim)
        # feature = self.layer_norm(feature)
        if B == 1:
            feature = feature.unsqueeze(0)

        # feature = self.suffix_bn(feature.permute(0,2,1)).permute(0,2,1)
        feature = self.suffix_linear(feature)
        return feature, N


class TLinear(nn.Module):
    def __init__(self, d_in, d_out, bias=True) -> None:
        super().__init__()
        self.linear_c = nn.Linear(d_in, d_out*d_in)
        self.W = nn.Parameter(torch.Tensor(d_in))
        self.mask = nn.Parameter(torch.Tensor(d_in, d_in))
        nn.init.kaiming_uniform_(self.mask, a=math.sqrt(5))
        # self.bn = nn.BatchNorm1d(d_in)
        nn.init.constant_(self.W, 1)
        self.bias = bias
        if bias:
            self.b = nn.Parameter(torch.Tensor(d_out))
            nn.init.constant_(self.b, 0)
        # self.channel_embedding = nn.Parameter(
        #     torch.from_numpy(get_1d_sincos_pos_embed_from_grid(d_model, np.arange(d_model))).float(), requires_grad=False)
        self.d_out = d_out
        self.d_in = d_in
        # self.pos_embedding = nn.Parameter(torch.Tensor(d_model))
        # nn.init.constant_(self.pos_embedding, 0.0)
        # self.diag = nn.Parameter(torch.diag(torch.ones(d_model)), requires_grad=False)
        # self.sigma = nn.Parameter(torch.tensor(0.0))
    # xyz: b x n x 3, features: b x n x f

    def forward(self, features):
        B, N, C = features.shape
        # features = self.bn(features.permute(0,2,1)).permute(0,2,1)
        # W = self.W/(torch.norm(self.W)+1e-5)*math.sqrt(self.d_in)
        # features *= W
        # relative_feature = features.unsqueeze(2) - features.unsqueeze(3)
        # relative_feature = relative_feature*self.mask
        # # make it sym
        # relative_feature = relative_feature * self.mask
        # relative_feature = relative_feature + relative_feature.permute(0, 1, 3, 2)
        #
        # relative_feature = torch.cat([relative_feature, features.unsqueeze(3).repeat(1, 1, 1, self.d_in)], -1)
        weight_c = self.linear_c(features).reshape(B, N, self.d_out, -1)
        weight_c = weight_c / (weight_c.abs() + 1e-5).sum(-2, keepdim=True)  #*math.sqrt(self.d_in)
        features = torch.matmul(weight_c, features.unsqueeze(2)).squeeze() # /math.sqrt(k.size(-1))
        return features


class MTLinear(nn.Module):
    def __init__(self, d_in, d_out, bias=True) -> None:
        super().__init__()
        self.linear_c = nn.Linear(d_in, d_out*d_in)
        self.W = nn.Parameter(torch.Tensor(d_in))
        self.mask = nn.Parameter(torch.Tensor(d_in, d_in))
        nn.init.kaiming_uniform_(self.mask, a=math.sqrt(5))
        # self.bn = nn.BatchNorm1d(d_in)
        nn.init.constant_(self.W, 1)
        self.bias = bias
        if bias:
            self.b = nn.Parameter(torch.Tensor(d_out))
            nn.init.constant_(self.b, 0)
        self.d_out = d_out
        self.d_in = d_in

    def forward(self, features):
        B, N, C = features.shape
        mfeature = features.mean(1)
        weight_c = self.linear_c(mfeature).reshape(B, self.d_out, -1)
        weight_c = weight_c / (weight_c.abs() + 1e-5).sum(-2, keepdim=True)  #*math.sqrt(self.d_in)
        features = torch.bmm(features, weight_c.permute(0, 2, 1))
        return features


class MLinear(nn.Module):
    def __init__(self, d_in, d_out, bias=True) -> None:
        super().__init__()
        self.linear_c = nn.Linear(d_in, d_out)
        self.W1 = nn.Parameter(torch.Tensor(d_in, d_in))
        self.W2 = nn.Parameter(torch.Tensor(d_in, d_in))
        nn.init.kaiming_uniform_(self.W1, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.W2, a=math.sqrt(5))
        self.d_out = d_out
        self.d_in = d_in

    def forward(self, features):
        B, N, C = features.shape
        mfeature = features.mean(dim=1)
        weight_c = mfeature.unsqueeze(1)*self.W1 - mfeature.unsqueeze(2)*self.W2
        weight_c = self.linear_c(weight_c) #.reshape(B, self.d_out, -1)
        weight_c = weight_c / (weight_c.abs() + 1e-5).sum(-1, keepdim=True)  #*math.sqrt(self.d_in)
        # features = torch.matmul(weight_c.unsqueeze(1), features.unsqueeze(3)).squeeze() # /math.sqrt(k.size(-1))
        features = torch.bmm(features, weight_c)
        return features


class TransformerBlock(nn.Module):
    def __init__(self, d_points, d_model, k=36) -> None:
        super().__init__()
        self.fc1 = nn.Linear(d_points, d_model)
        self.fc2 = nn.Linear(d_model, d_model)
        self.fc_delta = nn.Sequential(
            nn.Linear(3, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        self.fc_gamma = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        self.w_qs = nn.Linear(d_model, d_model, bias=False)
        self.w_ks = nn.Linear(d_model, d_model, bias=False)
        self.w_vs = nn.Linear(d_model, d_model, bias=False)
        self.k = k

    # xyz: b x n x 3, features: b x n x f
    def forward(self, features, xyz):
        dists = square_distance(xyz, xyz)
        knn_idx = dists.argsort()[:, :, :self.k]  # b x n x k
        knn_xyz = index_points(xyz, knn_idx)

        x = self.fc1(features)
        # q, k, v = self.w_qs(x), index_points(self.w_ks(x), knn_idx), index_points(self.w_vs(x), knn_idx)
        q, k, v = x, index_points(x, knn_idx), index_points(x, knn_idx)
        pos_enc = self.fc_delta(xyz[:, :, None] - knn_xyz)  # b x n x k x f

        attn = self.fc_gamma(q[:, :, None] - k + pos_enc)
        attn = F.softmax(attn / np.sqrt(k.size(-1)), dim=-2)  # b x n x k x f

        res = torch.einsum('bmnf,bmnf->bmf', attn, v + pos_enc )
        res = self.fc2(res) + x
        return res, attn


class CTransformerBlock(nn.Module):
    def __init__(self, d_points, d_model, k=36) -> None:
        super().__init__()
        self.fc1 = nn.Linear(d_points, d_model)
        self.fc2 =MTLinear(d_model, d_model)
        self.fc3 = MTLinear(d_model, d_model)
        self.bn = nn.BatchNorm1d(d_model)
        self.fc_delta = nn.Sequential(
            nn.Linear(3, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        self.fc_gamma = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        self.w_qs = nn.Linear(d_model, d_model, bias=False)
        self.w_ks = nn.Linear(d_model, d_model, bias=False)
        self.w_vs = nn.Linear(d_model, d_model, bias=False)
        self.k = k


    def forward(self, features, xyz):
        B, N, C = features.shape
        dists = square_distance(xyz, xyz)
        knn_idx = dists.argsort()[:, :, :self.k]  # b x n x k
        knn_xyz = index_points(xyz, knn_idx)

        x = self.fc1(features)
        q, k, v = self.w_qs(x), self.w_ks(x), self.w_vs(x)
        k = index_points(k, knn_idx)
        v = index_points(v, knn_idx)
        # k = k.unsqueeze(2).repeat(1,1,self.k, 1)
        # v = v.unsqueeze(2).repeat(1, 1, self.k, 1)
        # q, k, v = x, index_points(x, knn_idx), index_points(x, knn_idx)
        pos_enc = self.fc_delta(xyz[:, :, None] - knn_xyz)  # b x n x k x f

        attn = self.fc_gamma(q[:, :, None] - k + pos_enc)
        # attn = F.softmax(attn / np.sqrt(k.size(-1)), dim=-2)  # b x n x k x f
        # attn = attn / (attn.abs() + 1e-5).sum(-2, keepdim=True)
        attn = attn / (attn.abs() + 1e-5).sum(-1, keepdim=True)
        res = torch.einsum('bmnf,bmnf->bmf', attn, v + pos_enc)
        res = self.fc2(res) + x
        res = self.fc3(res) + res
        return res, attn



