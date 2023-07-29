import open3d as o3d
import torch
import numpy as np
import mcubes
from scipy import ndimage
import math


struct2 = ndimage.generate_binary_structure(3, 3)


textured_mesh = o3d.io.read_triangle_mesh("../ShapeNetCore.v1/02691156/fd0262e95283727d7b02bf6205401969/model.obj")
point = o3d.geometry.TriangleMesh.sample_points_uniformly(textured_mesh, number_of_points=2048)
voxel_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh(textured_mesh, voxel_size=1/5.0)

textured_mesh.compute_vertex_normals()
o3d.visualization.draw_geometries([textured_mesh, point], window_name="Open3D1")




def write_obj(obj_filename, vertices, triangles, output_max_component=True, scale=None):
    # normalize to -1,1
    resolution = 2 ** 7
    print("mesh resolution in {}".format(resolution))

    vertices /= (resolution / 2.0)
    vertices -= 1

    if (scale is not None):
        vertices *= scale

    with open(obj_filename, "w") as wf:
        for i in range(vertices.shape[0]):
            wf.write("v %lf %lf %lf\n" % (vertices[i][0], vertices[i][1], vertices[i][2]))
        for i in range(triangles.shape[0]):
            wf.write("f %d %d %d\n" % (triangles[i][0] + 1, triangles[i][1] + 1, triangles[i][2] + 1))


def marching_cubes(self, output_obj_filename, voxel_resolution):
    voxel_dim = 2 ** voxel_resolution
    voxel_length = 2.0 / voxel_dim
    voxel_dim += 1

    grids = np.ones([voxel_dim, voxel_dim,
                     voxel_dim]) * 1000000  # which means initial values are all not available (will be considered in our implemented marching_cubes_partial)
    assert (self.mls_radius is not None)

    mls_points_id = np.round((self.mls_points[:, :3] + 1) / voxel_length)

    np.clip(mls_points_id, 0, voxel_dim - 1, out=mls_points_id)
    mls_points_id = mls_points_id.astype(int)
    active_grids = np.zeros([voxel_dim, voxel_dim, voxel_dim])
    active_grids[mls_points_id[:, 0], mls_points_id[:, 1], mls_points_id[:, 2]] = 1.0;

    # might use larger dilation_layer_num
    max_radius = np.sqrt(np.max(self.mls_radius)) * 2
    dilation_layer_num = (int)(round(max_radius / voxel_length)) + 1
    print("dilation layer number: ", dilation_layer_num)
    active_grids_large = ndimage.binary_dilation(active_grids, structure=struct2, iterations=dilation_layer_num)

    nonzeros = np.nonzero(active_grids_large)
    evaluated_points = np.stack(nonzeros, axis=1) * voxel_length - 1
    print("active number: ", nonzeros[0].shape)

    print('outputfilename: ', output_obj_filename)
    vertices, triangles = mcubes.marching_cubes_partial(-grids, 0.0)
    write_obj(output_obj_filename, vertices, triangles, scale=self.scale)
# o3d.visualization.draw_geometries([voxel_grid])

if __name__ == "__main__":
    s=-0
    # marching_cubes("result.obj", 7)