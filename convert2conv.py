import os
import numpy as np
import torch
from tqdm import tqdm


def train_set():
    # sample_list = []
    # source_path = "/disk2/occupancy_networks-master/data/points2surf-master/datasets/abc_train/"
    # test_path = "/disk2/occupancy_networks-master/data/points2surf-master/datasets/abc_noisefree/"
    test_path = "/disk2/occupancy_networks-master/data/points2surf-master/datasets/famous_noisefree/"
    result_path = "/disk2/occupancy_networks-master/data/ABC_convert/001/"

    ## train
    # with open("/disk2/occupancy_networks-master/data/points2surf-master/datasets/abc_train/trainset.txt", "r") as f:
    #     sample_list_all = f.readlines()
    # sample_list = [x[0:-1] for x in sample_list_all]
    # # sample_list.append(sample_list_all[-1])
    # sss= sample_list[-1]
    # with open(result_path + "train.lst", "w") as f:
    #     for idx in tqdm(sample_list):
    #         code = idx[0:8]
    #         pointcloud = np.load(source_path+"04_pts/"+idx+".xyz.npy")
    #         # choice = torch.randperm(pointcloud.shape[0])[0:3000].numpy()
    #         # pointcloud = pointcloud[choice]
    #         query_sdf = np.clip(np.load(source_path + "05_query_dist/" + idx + ".ply.npy"), a_max=0.05, a_min=-0.05)
    #         query_p = np.load(source_path + "05_query_pts/" + idx + ".ply.npy")
    #         query_l = np.packbits(((np.sign(query_sdf) + 1) / 2).astype(np.int))
    #         if not os.path.exists(result_path+code):
    #             os.makedirs(result_path+code)
    #         np.savez(result_path+code+"/points.npz", points=query_p, occupancies=query_l, sdf=query_sdf)
    #         np.savez(result_path+code+"/pointcloud.npz", points=pointcloud)
    #         f.write(code+"\n")
    # # test
    with open("/disk2/occupancy_networks-master/data/points2surf-master/datasets/famous_noisefree/testset.txt", "r") as f:
        test_list = f.readlines()
    sample_list = [x[0:-1] for x in test_list[0:-1]]
    sample_list.append(test_list[-1])
    with open(result_path + "test.lst", "w") as f:
        for idx in tqdm(sample_list):
            code = idx[0:8]
            pointcloud = np.load(test_path + "04_pts/" + idx + ".xyz.npy")
            choice = torch.randperm(pointcloud.shape[0])[0:3000].numpy()
            pointcloud = pointcloud[choice]
            if not os.path.exists(result_path + code):
                os.makedirs(result_path + code)
            normals = np.ones_like(pointcloud)/3.0*1.73
            np.savez(result_path + code + "/pointcloud.npz", points=pointcloud, normals=normals)
            f.write(code + "\n")
    ## val
    # with open("/disk2/occupancy_networks-master/data/points2surf-master/datasets/abc_train/valset.txt", "r") as f:
    #     sample_list_all = f.readlines()
    # sample_list = [x[0:-1] for x in sample_list_all]
    # # sample_list.append(sample_list_all[-1])
    # with open(result_path + "val.lst", "w") as f:
    #     for idx in tqdm(sample_list):
    #         code = idx[0:8]
    #         pointcloud = np.load(source_path + "04_pts/" + idx + ".xyz.npy")
    #         choice = torch.randperm(pointcloud.shape[0])[0:3000].numpy()
    #         pointcloud = pointcloud[choice]
    #         query_sdf = np.clip(np.load(source_path + "05_query_dist/" + idx + ".ply.npy"), a_max=0.05, a_min=-0.05)
    #         query_p = np.load(source_path + "05_query_pts/" + idx + ".ply.npy")
    #         query_l = np.packbits(((np.sign(query_sdf) + 1) / 2).astype(np.int))
    #         if not os.path.exists(result_path + code):
    #             os.makedirs(result_path + code)
    #         np.savez(result_path + code + "/points.npz", points=query_p, occupancies=query_l, sdf=query_sdf)
    #         normals = np.ones_like(pointcloud)/3.0*1.73
    #         np.savez(result_path + code + "/pointcloud.npz", points=pointcloud, normals=normals)
    #         f.write(code + "\n")


def test_in_out():
    point = np.load("/disk2/occupancy_networks-master/data/points2surf-master/datasets/abc_train/04_pts/00010071_493cf58028d24a5b97528c11_trimesh_001.xyz.npy")
    dist = np.load("/disk2/occupancy_networks-master/data/points2surf-master/datasets/abc_train/05_query_dist/00010071_493cf58028d24a5b97528c11_trimesh_001.ply.npy")
    query = np.load("/disk2/occupancy_networks-master/data/points2surf-master/datasets/abc_train/05_query_pts/00010071_493cf58028d24a5b97528c11_trimesh_001.ply.npy")
    choice = torch.randperm(point.shape[0])[0:3000].numpy()
    point = point[choice]
    np.savetxt("pp.txt", point, delimiter=";")
    np.savetxt("query.txt", query, delimiter=";")
    ss = 0


if __name__ == '__main__':
    train_set()
    # test_in_out()
    # point = np.load("/disk2/occupancy_networks-master/data/points2surf-master/datasets/abc_train/04_pts/00010071_493cf58028d24a5b97528c11_trimesh_001.xyz.npy")
    # dist = np.load("/disk2/occupancy_networks-master/data/points2surf-master/datasets/abc_train/05_query_dist/00010006_7e4956ae07e24f6584127385_trimesh_000.ply.npy")
    # query = np.load("/disk2/occupancy_networks-master/data/points2surf-master/datasets/abc_train/05_query_pts/00010006_7e4956ae07e24f6584127385_trimesh_000.ply.npy")
    # np.savetxt("pp.txt", point, delimiter=";")
    # ss = 0

