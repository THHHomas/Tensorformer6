import open3d as o3d
import torch
import os
import numpy as np


def ball_pivoting(point_cloud_path):
    pc = np.load(point_cloud_path)
    point_cloud, normal = pc["points"], pc["normals"]
    point_choice = torch.randperm(point_cloud.shape[0])[0:3000].numpy()
    point_cloud, normal = point_cloud[point_choice], normal[point_choice]
    point_cloud = np.random.randn(*point_cloud.shape)*0.005+point_cloud
    # normal = np.random.randn(*normal.shape) * 0.1 + normal
    normal = normal/(1e-7+ np.linalg.norm(normal, axis=-1, keepdims=True))
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud)
    pcd.normals = o3d.utility.Vector3dVector(normal)
    # o3d.visualization.draw_geometries([pcd], point_show_normal=True)

    radii = [0.005, 0.01, 0.02, 0.04]
    rec_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        pcd, o3d.utility.DoubleVector(radii))
    # rec_mesh.compute_vertex_normals()
    # o3d.visualization.draw_geometries([rec_mesh])
    return rec_mesh


def poisson(point_cloud_path):
    pc = np.load(point_cloud_path)
    point_cloud, normal = pc["points"], pc["normals"]
    point_choice = torch.randperm(point_cloud.shape[0])[0:3000].numpy()
    point_cloud, normal = point_cloud[point_choice], normal[point_choice]
    point_cloud = np.random.randn(*point_cloud.shape) * 0.005 + point_cloud
    normal = np.random.randn(*normal.shape) * 0.0 + normal
    normal = normal / (1e-7 + np.linalg.norm(normal, axis=-1, keepdims=True))

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud)
    pcd.normals = o3d.utility.Vector3dVector(normal)
    # o3d.visualization.draw_geometries([pcd])
    # pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(0.1, 32))

    print('run Poisson surface reconstruction')
    # with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)
    # mesh.compute_vertex_normals()
    # mesh.paint_uniform_color([229/255,231/255,1.0])
    # o3d.visualization.draw_geometries([mesh])
    return mesh


# def poisson(mesh_path):
#     gt_mesh = o3d.io.read_triangle_mesh(mesh_path)
#     gt_mesh.compute_vertex_normals()
#     pcd = gt_mesh.sample_points_poisson_disk(3000)
#     # points, face_idx = gt_mesh.sample(3000, return_index=True)
#
#     # o3d.visualization.draw_geometries([pcd])
#
#     print('run Poisson surface reconstruction')
#     # with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
#     mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)
#     # mesh.compute_vertex_normals()
#     # mesh.paint_uniform_color([229/255,231/255,1.0])
#     # o3d.visualization.draw_geometries([mesh])
#     return mesh


def poisson_test(path):
    # point_root_path = "/disk2/occupancy_networks-master/data/ABC/001/"  # 00015002/pointcloud.npz
    # test_file_path = "/disk2/occupancy_networks-master/data/ABC/001/test.lst"
    point_root_path = "/disk2/occupancy_networks-master/data/ShapeNet/04530566/"  # 00015002/pointcloud.npz
    test_file_path = "/disk2/occupancy_networks-master/data/ShapeNet/04530566/test.lst"
    with open(test_file_path, "r") as f:
        idx_list = f.readlines()
    idx_list = [x[0:-1] for x in idx_list if x[-1] == "\n"]
    test_list = idx_list
    # test_list = sorted(idx_list)
    for idx in test_list:
        pc_path = point_root_path + idx+ "/pointcloud.npz"
        mesh = poisson(pc_path)
        o3d.io.write_triangle_mesh(path+"/"+idx+"_psr.off", mesh)


def bpa_test(path):
    point_root_path = "/disk2/occupancy_networks-master/data/ABC/001/"  # 00015002/pointcloud.npz
    test_file_path = "/disk2/occupancy_networks-master/data/ABC/001/test.lst"
    with open(test_file_path, "r") as f:
        idx_list = f.readlines()
    idx_list = [x[0:-1] for x in idx_list if x[-1] == "\n"]

    test_list = sorted(idx_list)
    for idx in test_list:
        pc_path = point_root_path + idx + "/pointcloud.npz"
        mesh = ball_pivoting(pc_path)
        o3d.io.write_triangle_mesh(path + "/" + idx + "_bpa.off", mesh)


if __name__ == '__main__':
    # poisson_test("samples/bsp_ae_out/001/PSR")
    # bpa_test("samples/bsp_ae_out/001/BPA")
    poisson_test("samples/bsp_ae_out/04530566/PSR")

