import trimesh
import open3d as o3d
import numpy as np

# Load a surface to voxelize
filename = "/home/magician/ShapeNetCore.v1/02691156/144070f63941362bd1810447902e2904/model.obj"# "/home/magician/BSP-NET-pytorch/samples/bsp_ae_out/3_vox.ply"

# trimesh
mesh = trimesh.load(filename)
ss = trimesh.__version__
vv = trimesh.voxel.creation.voxelize(mesh, 1/64.0)
s = 0


# open3d
# mesh = o3d.io.read_triangle_mesh(filename)
# voxel_mesh = o3d.geometry.VoxelGrid.create_from_triangle_mesh(mesh, 1 / 256.0)
# o3d.visualization.draw_geometries([mesh, voxel_mesh])

