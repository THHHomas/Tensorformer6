import torch
import numpy as np
import os
import trimesh
import open3d as o3d
from libkdtree import KDTree
from utils import sample_points, read_ply_polygon, write_ply_polygon
from mesh_intersection import check_mesh_contains, compute_iou


def as_mesh(scene_or_mesh):
    """
    Convert a possible scene to a mesh.

    If conversion occurs, the returned mesh has only vertex and face data.
    """
    if isinstance(scene_or_mesh, trimesh.Scene):
        if len(scene_or_mesh.geometry) == 0:
            mesh = None  # empty scene
        else:
            # we lose texture information here
            mesh = trimesh.util.concatenate(
                tuple(trimesh.Trimesh(vertices=g.vertices, faces=g.faces)
                    for g in scene_or_mesh.geometry.values()))
    else:
        assert(isinstance(scene_or_mesh, trimesh.Trimesh))
        mesh = scene_or_mesh
    return mesh


def l2_distance(src, dst):
    """
    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist.sqrt()


def l1_distance(src, dst):
    """
    Calculate l1 distance between each two points.

        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    dist = (src.unsqueeze(2) - dst.unsqueeze(1)).abs().sum(-1)
    return dist


def charmer_distance(pc1, pc2):
    cloeset_dis = l1_distance(pc1[:,:,0:3], pc2[:,:,0:3])
    ver_closest_dis, ver_closest_idx = cloeset_dis.min(dim=-2)
    cloeset_dis, cloeset_idx = cloeset_dis.min(dim=-1)

    normal1 = pc1[:, :,3:]
    normal2 = pc2[:, cloeset_idx.squeeze(), 3:]
    nc = torch.matmul(normal1.unsqueeze(2), normal2.unsqueeze(3)).squeeze().unsqueeze(0)

    normal1 = pc1[:, ver_closest_idx.squeeze(), 3:]
    normal2 = pc2[:, :, 3:]
    nc_ver = torch.matmul(normal1.unsqueeze(2), normal2.unsqueeze(3)).squeeze().unsqueeze(0)
    ss = nc_ver.mean()
    ss2 = nc.mean()
    nc = (ss + ss2) / 2
    cd = (cloeset_dis.mean() + ver_closest_dis.mean()) / 2
    return cd, nc


def distance_p2p(points_src, normals_src, points_tgt, normals_tgt):
    ''' Computes minimal distances of each point in points_src to points_tgt.

    Args:
        points_src (numpy array): source points
        normals_src (numpy array): source normals
        points_tgt (numpy array): target points
        normals_tgt (numpy array): target normals
    '''
    kdtree = KDTree(points_tgt)
    dist, idx = kdtree.query(points_src)
    if normals_src is not None and normals_tgt is not None:
        normals_src = \
            normals_src / np.linalg.norm(normals_src, axis=-1, keepdims=True)
        normals_tgt = \
            normals_tgt / np.linalg.norm(normals_tgt, axis=-1, keepdims=True)

        normals_dot_product = (normals_tgt[idx] * normals_src).sum(axis=-1)
        # Handle normals that point into wrong direction gracefully
        # (mostly due to mehtod not caring about this in generation)
        normals_dot_product = np.abs(normals_dot_product)
    else:
        normals_dot_product = np.array(
            [np.nan] * points_src.shape[0], dtype=np.float32)
    return dist, normals_dot_product


def evaluate_cd_nc(path, test_file):
    obj_root_path = "../ShapeNetCore.v1/"
    test_file = open(test_file, "r")
    chosen_suffix = "_vox.ply"
    test_file = test_file.readlines()
    cd_total = 0
    iou_total = 0
    nc_total = 0
    iou_list = []
    nc_list = []
    pc1 = []
    pc2 = []
    idx_list = []
    for ff in os.listdir(path):
        idx = ff.split("_")[0]
        if eval(idx) not in idx_list:
            idx_list.append(eval(idx))
    sample_num = 0
    test_list = sorted(idx_list)
    for idx in test_list:
        idx = str(idx)
        print(idx)

        # vertices, triangles = np.asarray(mesh.vertices), np.asarray(mesh.triangles)
        # vertices[:, [0, 2]] = vertices[:, [2, 0]]
        # pc_gt = sample_points(vertices, triangles, 100000)
        mesh = o3d.io.read_triangle_mesh(path + "/" + str(idx) + "_gt.ply")
        # mesh = mesh.filter_smooth_laplacian(number_of_iterations=1)
        vertices, polygons = np.asarray(mesh.vertices), np.asarray(mesh.triangles)
        # vertices, polygons = read_ply_polygon(path + "/" + str(idx) + "_vox.ply")
        pc_vox = sample_points(vertices, polygons, 100000)

        pc1.append(pc_vox)

        mesh = o3d.io.read_triangle_mesh(path + "/" + str(idx) + chosen_suffix)
        # mesh = mesh.filter_smooth_laplacian(number_of_iterations=3)
        vertices, polygons = np.asarray(mesh.vertices), np.asarray(mesh.triangles)
        # vertices, polygons = read_ply_polygon(path + "/" + str(idx) + "_gt.ply")
        # vertices[:,[0,2]] = vertices[:,[2,0]]
        # mesh = o3d.geometry.TriangleMesh()
        # mesh.vertices = o3d.utility.Vector3dVector(vertices)
        # mesh.triangles = o3d.utility.Vector3iVector(polygons)
        # o3d.visualization.draw_geometries([mesh, mesh1])

        pc_gt = sample_points(vertices, polygons, 100000)
        pc2.append(pc_gt)
        # np.savetxt("cccc/"+str(idx)+"_gt.txt", pc_gt)
        # np.savetxt("cccc/"+str(idx)+"_vox.txt", pc_vox)

        pc1 = torch.tensor(pc1)
        pc2 = torch.tensor(pc2)
        pc1 = pc1.numpy()
        pc2 = pc2.numpy()


        pointcloud_tgt = pc2[0,:, 0:3]
        normals_tgt = pc2[0,:,3:]
        pointcloud = pc1[0,:,0:3]
        normals = pc1[0,:,3:]
        completeness, completeness_normals = distance_p2p(
            pointcloud_tgt, normals_tgt, pointcloud, normals
        )
        accuracy, accuracy_normals = distance_p2p(
            pointcloud, normals, pointcloud_tgt, normals_tgt
        )
        completeness = completeness.mean()
        completeness_normals = completeness_normals.mean()
        accuracy = accuracy.mean()
        accuracy_normals = accuracy_normals.mean()
        chamferL1 = 0.5 * (completeness + accuracy)
        normals_correctness = (
                0.5 * completeness_normals + 0.5 * accuracy_normals
        )

        # chamferL1, normals_correctness = charmer_distance(pc1, pc2)
        cd_total += chamferL1
        nc_total += normals_correctness
        nc_list.append(normals_correctness)
        pc1 = []
        pc2 = []

        #  calculate iou
        # points_iou = np.load("points.npy")
        # np.savetxt("test.txt", points_iou)
        sample_size_expand = 64+8
        xyz_grid = torch.meshgrid([torch.arange(0, sample_size_expand), torch.arange(0, sample_size_expand), torch.arange(0, sample_size_expand)])
        sample_size = 64
        points_iou = torch.stack([xyz_grid[0].reshape(-1)/float(sample_size), xyz_grid[1].reshape(-1)/float(sample_size),
                                  xyz_grid[2].reshape(-1)/float(sample_size)]).permute(1,0).numpy()-float(sample_size_expand/2)/float(sample_size)
        rand_shift = torch.rand([sample_size_expand*sample_size_expand*sample_size_expand, 3]).numpy()*0.0001
        points_iou += rand_shift
        # np.savetxt("test1.txt", points_iou)

        mesh = trimesh.load(path + "/" + str(idx) + "_gt.ply", process=False)
        # mesh = as_mesh(mesh)
        # mesh.show()
        # sss= trimesh.sample.volume_mesh(mesh, 1000)
        occ = check_mesh_contains(mesh, points_iou)
        mesh = trimesh.load(path + "/" + str(idx) + chosen_suffix, process=False)
        chosen_data = points_iou[np.where(occ)]
        # np.savetxt("test1.txt", chosen_data)

        occ2 = check_mesh_contains(mesh, points_iou)
        # ss, tt = occ.sum(), occ2.sum()
        # chosen_data = points_iou[np.where(occ2)]
        # np.savetxt("test2.txt", chosen_data)
        iou = compute_iou(occ, occ2)
        iou_total += iou
        iou_list.append(iou)
        sample_num += 1

        print("finished  batch ", float(sample_num)/len(test_list))
    print(nc_list)
    print(iou_list)
    cd_total = cd_total/float(sample_num)
    nc_total = nc_total / float(sample_num)
    iou_total = iou_total / float(sample_num)
    return cd_total, nc_total, iou_total


def evaluate_cd_nc_abc(path, original_mehs_path, topk = 3, chosen_suffix=".off", log_file="abc_normat"):
    cd_total = 0
    iou_total = 0
    nc_total = 0
    cd_list = []
    nc_list = []
    pc1 = []
    pc2 = []
    # idx_list = []
    # for ff in os.listdir(path):
    #     idx, suffix = ff.split(".")
    #     if suffix == "off" and len(idx) == 8:
    #         idx_list.append(idx)
    test_file_path = "/disk2/occupancy_networks-master/data/ABC/001/test.lst"
    with open(test_file_path, "r") as f:
        idx_list = f.readlines()
    idx_list = [x[0:-1] for x in idx_list  if x[-1] =="\n"]
    sample_num = 0
    test_list = sorted(idx_list)  #[749:752]
    # test_list = test_list[int(0.6*len(test_list)):]
    # print(len(test_list))
    # exit(0)
    for idx in test_list:
        idx = str(idx)
        print(idx)
        mesh = o3d.io.read_triangle_mesh(path + "/" + str(idx) + chosen_suffix)
        # mesh = mesh.filter_smooth_laplacian(number_of_iterations=1)
        vertices, polygons = np.asarray(mesh.vertices), np.asarray(mesh.triangles)
        # vertices, polygons = read_ply_polygon(path + "/" + str(idx) + "_vox.ply")
        pc_vox = sample_points(vertices, polygons, 100000)

        pc1=[pc_vox]

        mesh = o3d.io.read_triangle_mesh(original_mehs_path + "/" + str(idx) + ".off")
        # mesh = mesh.filter_smooth_laplacian(number_of_iterations=3)
        vertices, polygons = np.asarray(mesh.vertices), np.asarray(mesh.triangles)
        pc_gt = sample_points(vertices, polygons, 100000)
        pc2 = [pc_gt]
        pc1 = torch.tensor(pc1)
        pc2 = torch.tensor(pc2)
        pc1 = pc1.numpy()
        pc2 = pc2.numpy()


        pointcloud_tgt = pc2[0,:, 0:3]
        normals_tgt = pc2[0,:,3:]
        pointcloud = pc1[0,:,0:3]
        normals = pc1[0,:,3:]
        completeness, completeness_normals = distance_p2p(
            pointcloud_tgt, normals_tgt, pointcloud, normals
        )
        accuracy, accuracy_normals = distance_p2p(
            pointcloud, normals, pointcloud_tgt, normals_tgt
        )
        completeness = completeness.mean()
        completeness_normals = completeness_normals.mean()
        accuracy = accuracy.mean()
        accuracy_normals = accuracy_normals.mean()
        chamferL1 = 0.5 * (completeness + accuracy)
        normals_correctness = (
                0.5 * completeness_normals + 0.5 * accuracy_normals
        )

        # chamferL1, normals_correctness = charmer_distance(pc1, pc2)
        cd_total += chamferL1
        nc_total += normals_correctness
        nc_list.append(normals_correctness)
        cd_list.append(chamferL1)


        #########################################################################
        sample_size_expand = 64 + 8
        xyz_grid = torch.meshgrid([torch.arange(0, sample_size_expand), torch.arange(0, sample_size_expand),
                                   torch.arange(0, sample_size_expand)])
        sample_size = 64
        points_iou = torch.stack(
            [xyz_grid[0].reshape(-1) / float(sample_size), xyz_grid[1].reshape(-1) / float(sample_size),
             xyz_grid[2].reshape(-1) / float(sample_size)]).permute(1, 0).numpy() - float(
            sample_size_expand / 2) / float(sample_size)
        rand_shift = torch.rand([sample_size_expand * sample_size_expand * sample_size_expand, 3]).numpy() * 0.0005
        points_iou += rand_shift
        # np.savetxt("test1.txt", points_iou)

        mesh = trimesh.load(original_mehs_path + "/" + str(idx) + ".off", process=False)
        # mesh = as_mesh(mesh)
        # mesh.show()
        # sss= trimesh.sample.volume_mesh(mesh, 1000)
        occ = check_mesh_contains(mesh, points_iou)
        mesh = trimesh.load(path + "/" + str(idx) + chosen_suffix, process=False)
        # chosen_data = points_iou[np.where(occ)]
        # np.savetxt("test1.txt", chosen_data)

        occ2 = check_mesh_contains(mesh, points_iou)
        # ss, tt = occ.sum(), occ2.sum()
        # chosen_data = points_iou[np.where(occ2)]
        # np.savetxt("test2.txt", chosen_data)
        iou = compute_iou(occ, occ2)
        if np.isnan(iou):
            iou = 0
        iou_total += iou
        # iou_list.append(iou)
        ###########################################################
        sample_num += 1

        print("finished  batch ", float(sample_num)/len(test_list))
    # print(nc_list)
    # print(cd_list)

    nc_top_k = np.argsort(nc_list)[0:topk]
    nc_bottom_k = np.argsort(nc_list)[::-1][0:topk]

    cd_top_k = np.argsort(cd_list)[0:topk]
    cd_bottom_k = np.argsort(cd_list)[::-1][0:topk]

    cd_total = cd_total/float(sample_num)
    nc_total = nc_total / float(sample_num)
    iou_total = iou_total / float(sample_num)

    with open("samples/bsp_ae_out/001/"+log_file+".log", "w") as f:
        f.write("cd is %f:"%cd_total.item()+" nc is: %f"%nc_total.item()+" iou is: %f\n"%iou_total.item())
        f.write("cd top k:\n")
        for i in cd_top_k:
            f.write(test_list[i]+"\n")

        f.write("cd bottom k:\n")
        for i in cd_bottom_k:
            f.write(test_list[i] + "\n")

        f.write("nc top k:\n")
        for i in nc_top_k:
            f.write(test_list[i] + "\n")

        f.write("nc bottom k:\n")
        for i in nc_bottom_k:
            f.write(test_list[i] + "\n")

    return cd_total, nc_total


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # test_file = "data/data_per_category/data_per_category/02691156_airplane/02691156_vox256_img_test.txt"
    # test_file = "data/data_per_category/data_per_category/00000000_all/00000000_vox256_img_test.txt"
    # cd, nc, iou = evaluate_cd_nc("samples/bsp_ae_out", test_file)

    cd, nc = evaluate_cd_nc_abc("/disk2/occupancy_networks-master/data/points2surf-master/results/p2s_max_model_249/abc_001/rec/mesh/",
                                "/disk2/occupancy_networks-master/data/ABC.build/001/4_watertight_scaled/",
                                chosen_suffix=".ply", log_file="abc_p2s")
    print("cd is:", cd.item(), " nc is:", nc.item())
    # cd, nc = evaluate_cd_nc_abc(
    #     "/home/magician/convolutional_occupancy_networks-master/out/pointcloud/shapenet_grid32/generation_pretrained/meshes/001/",
    #     "/disk2/occupancy_networks-master/data/ABC.build/001/4_watertight_scaled/",
    #     chosen_suffix=".off", log_file="abc_convocc")
    cd, nc = evaluate_cd_nc_abc("samples/bsp_ae_out/001",
                                "/disk2/occupancy_networks-master/data/ABC.build/001/4_watertight_scaled/", chosen_suffix=".off", log_file="abc_normat")
    # cd, nc = evaluate_cd_nc_abc("samples/bsp_ae_out/001/PSR",
    #                             "/disk2/occupancy_networks-master/data/ABC.build/001/4_watertight_scaled/", chosen_suffix="_psr.off", log_file="abc_psr")
    # cd, nc = evaluate_cd_nc_abc("samples/bsp_ae_out/001/BPA",
    #                             "/disk2/occupancy_networks-master/data/ABC.build/001/4_watertight_scaled/", chosen_suffix="_bpa.off", log_file="abc_bpa")

    print("cd is:", cd.item(), " nc is:", nc.item())