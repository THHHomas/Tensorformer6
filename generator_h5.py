import h5py
import numpy as np
import os
import open3d as o3d
import shutil
import torch
import mcubes
from utils import sample_points
from utils import write_ply_triangle
import random
from utils import farthest_point_sample


def generator_h5(sample_txt_file, mesh_path, phase):
    # ff = set()
    txt_file = open(sample_txt_file, "r")
    point_cloud_list = []
    point_idx_list = []
    point_num = 3000
    txt_file = txt_file.readlines()
    prefix, _ = os.path.split(sample_txt_file)
    with h5py.File(os.path.join(prefix, phase + ".h5"), "w") as F:
        for idx, target_file in enumerate(txt_file):
            # ff.add(target_file[0:8])
            if os.path.exists(mesh_path+target_file[:-1]+"/images"):
                shutil.rmtree(mesh_path+target_file[:-1]+"/images")
            current_mesh_path = mesh_path+target_file[:-1]+"/model.obj"
            mesh = o3d.io.read_triangle_mesh(current_mesh_path)
            vertices, triangles = np.asarray(mesh.vertices), np.asarray(mesh.triangles)
            pc = sample_points(vertices, triangles, point_num)

            point_cloud_list.append(pc)
            print(idx, float(idx)/len(txt_file))
        point_cloud_list = np.stack(point_cloud_list)
        F.create_dataset("point", data=point_cloud_list)


def sample_from_watertight_mesh(sample_txt_file, mesh_path, phase, reverse):
    txt_file = open(sample_txt_file, "r")
    point_cloud_list = []
    point_num = 4096
    txt_file = txt_file.readlines()
    prefix, _ = os.path.split(sample_txt_file)
    data_hdf5_name = sample_txt_file.split(".")[0]+".hdf5"
    batch_voxels_ = h5py.File(data_hdf5_name, 'r')['voxels'][:].squeeze()
    with h5py.File(os.path.join(prefix, phase + "4096.h5"), "w") as F:
        for idx, target_file in enumerate(txt_file):
            # ff.add(target_file[0:8])

            vertices, triangles = mcubes.marching_cubes(batch_voxels_[idx, :, :, :], 0.5)
            vertices = (vertices + 0.5) / 64 - 0.5
            # output ground truth
            mesh = o3d.geometry.TriangleMesh()
            mesh.vertices = o3d.utility.Vector3dVector(vertices)
            mesh.triangles = o3d.utility.Vector3iVector(triangles)
            vertices, triangles = np.asarray(mesh.vertices), np.asarray(mesh.triangles)
            pc = sample_points(vertices, triangles, point_num)

            # pcd = o3d.geometry.PointCloud()
            # pcd.points = o3d.utility.Vector3dVector(pc)
            # octree = o3d.geometry.Octree(max_depth=4)
            # # 从点云中构建八叉树，适当扩展边界0.01m
            # octree.convert_from_point_cloud(pcd, size_expand=0.01)
            # # 可视化
            # o3d.visualization.draw_geometries([octree])
            #
            # voxel_idx = np.stack(np.where(batch_voxels_[idx, :, :, :]==1)).transpose(1, 0)/64.0-0.5+0.5/64
            # write_ply_triangle("gt.ply", vertices, triangles)
            # np.savetxt("pc.txt", pc[:,0:3])
            # np.savetxt("bv.txt", voxel_idx)
            # if reverse:
            #     pc[:, [0,2]] = pc[:, [2,0]]
            #     pc[:, 2] = -pc[:, 2]
            point_cloud_list.append(pc)

            print(idx, float(idx) / len(txt_file))
        point_cloud_list = np.stack(point_cloud_list)

        F.create_dataset("point", data=point_cloud_list)


if __name__ == "__main__":
    for cls in ["02691156_airplane"]:  #["03001627_chair"]: # ["02691156_airplane"]:#, "04530566_vessel", "04379243_table"]:
        cls_pre = cls.split("_")[0]
        for phase in ["train", "test"]:
            sample_from_watertight_mesh("data/data_per_category/data_per_category/"+cls+"/"+cls_pre+"_vox256_img_"+phase+".txt", mesh_path="/home/magician/ShapeNetCore.v1/", phase=phase, reverse=False)
    # for phase in ["train", "test"]:
    #     sample_from_watertight_mesh("data/data_per_category/data_per_category/00000000_all/"+"00000000_vox256_img_"+phase+".txt", mesh_path="/home/magician/ShapeNetCore.v1/", phase=phase, reverse=True)
