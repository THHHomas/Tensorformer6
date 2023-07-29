import open3d as o3d
import os
from utils import write_ply_polygon
import numpy as np

def ball_pivoting(mesh_path):
    gt_mesh = o3d.io.read_triangle_mesh(mesh_path)
    gt_mesh.compute_vertex_normals()
    pcd = gt_mesh.sample_points_poisson_disk(3000)
    # o3d.visualization.draw_geometries([pcd], point_show_normal=True)

    radii = [0.005, 0.01, 0.02, 0.04]
    rec_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
                   pcd, o3d.utility.DoubleVector(radii))
    # rec_mesh.compute_vertex_normals()
    # o3d.visualization.draw_geometries([rec_mesh])
    return rec_mesh


def poisson(mesh_path):
    gt_mesh = o3d.io.read_triangle_mesh(mesh_path)
    gt_mesh.compute_vertex_normals()
    pcd = gt_mesh.sample_points_poisson_disk(3000)

    # o3d.visualization.draw_geometries([pcd])

    print('run Poisson surface reconstruction')
    # with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)
    # mesh.compute_vertex_normals()
    # mesh.paint_uniform_color([229/255,231/255,1.0])
    # o3d.visualization.draw_geometries([mesh])
    return mesh

def poisson_test(path):
    idx_list = []
    for ff in os.listdir(path):
        idx = ff.split("_")[0]
        if eval(idx) not in idx_list:
            idx_list.append(eval(idx))
    sample_num = 0
    test_list = sorted(idx_list)[0:16]
    for idx in test_list:
        mesh_path = path+"/"+str(idx)+"_gt.ply"
        mesh = poisson(mesh_path)
        vertices, polygons = np.asarray(mesh.vertices), np.asarray(mesh.triangles)
        write_ply_polygon(path+"/"+str(idx)+"_psr.ply", vertices, polygons)




def bpa_test(path):
    idx_list = []
    for ff in os.listdir(path):
        idx = ff.split("_")[0]
        if eval(idx) not in idx_list:
            idx_list.append(eval(idx))
    sample_num = 0
    test_list = sorted(idx_list)[0:16]
    for idx in test_list:
        mesh_path = path+"/"+str(idx)+"_gt.ply"
        mesh = ball_pivoting(mesh_path)
        vertices, polygons = np.asarray(mesh.vertices), np.asarray(mesh.triangles)
        write_ply_polygon(path+"/"+str(idx)+"_bpa.ply", vertices, polygons)

if __name__ == '__main__':

    poisson_test("samples/bsp_ae_out")
    bpa_test("samples/bsp_ae_out")