import open3d as o3d
import numpy as np
import torch
import os
import shutil
import h5py
import random
from mesh_to_sdf import mesh_to_voxels
from mesh_to_sdf import sample_sdf_near_surface
import pyrender

import trimesh
import skimage
import mcubes


def generate_point_h5(file_prefix, class_dir, phase="train"):
    train_file = file_prefix + "_" + phase + ".txt"
    train_list = open(class_dir+train_file, "r").read().split("\n")
    point_cloud_list = []
    point_file = class_dir+phase+'.h5'
    # dire = "class_dir/point"
    for idx, mesh_file in enumerate(train_list):
        print(mesh_file)
        # mesh_file = "04530566/cd4240b8246555df54102e7ecaeb1c5"
        textured_mesh = o3d.io.read_triangle_mesh("../ShapeNetCore.v1/" + mesh_file + '/model.obj')
        point = o3d.geometry.TriangleMesh.sample_points_uniformly(textured_mesh, number_of_points=1024)
        point_origin = np.array(point.points)
        point = point_origin + 0.5

        point_cloud_list.append(point)
        if idx % 300==0:
            print("finieshed: ", float(idx)/len(train_list))
    with h5py.File(point_file, "w") as F:
        point_cloud_list = np.stack(point_cloud_list)
        F.create_dataset("point", data=point_cloud_list)


def generator_point(sample_txt_file, box_path, phase):
    txt_file = open(sample_txt_file, "r")
    point_cloud_list = []
    box_list = []
    point_num = 1024
    txt_file = txt_file.readlines()
    prefix, _ = os.path.split(sample_txt_file)
    with h5py.File(os.path.join(prefix, phase + ".h5"), "w") as F:
        for idx, target_file in enumerate(txt_file):
            point_path = box_path+target_file[:-1]+"/points.txt"
            point = np.loadtxt(point_path, delimiter=' ', dtype=np.float32)[:, 0:3]
            box = np.loadtxt(box_path+target_file[:-1]+"/box.txt", delimiter=' ', dtype=np.float32)
            N = point.shape[0]
            if N >= point_num:
                point = point[0:point_num]
            else:
                point_else = random.choices(point, k=point_num-N)
                point = np.concatenate([point, point_else], 0)
            # point = pc_normalize_n(point)
            point_cloud_list.append(point)
            box_list.append(box)

        point_cloud_list = np.stack(point_cloud_list)
        box_list = np.stack(box_list)

        F.create_dataset("box", data=box_list)
        F.create_dataset("point", data=point_cloud_list)


def remove_images():
    ss = os.walk("../ShapeNetCore.v1/")
    for x in ss:

        if x[1] == ['images']:
            print(x)
            shutil.rmtree(x[0] + "/" + x[1][0])
    exit(0)


if __name__ == "__main__":
    # remove_images()
    class_dir = "data/data/all_vox256_img/"
    file_prefix = "all_vox256_img"
    generate_point_h5(file_prefix, class_dir, phase="train")
    generate_point_h5(file_prefix, class_dir, phase="test")
