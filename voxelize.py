#!/usr/bin/env python

""" Compute voxelization of an input mesh.
"""

import argparse
import pymesh
import trimesh
from mesh_to_sdf import sample_sdf_near_surface
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__);
    parser.add_argument("--cell-size", default=1/64.0, type=float,
            help="size of each voxel");
    parser.add_argument("--erode", help="erode N layers of voxel", default=0,
            type=int, metavar="N");
    parser.add_argument("--dilate", help="dilate N layers of voxel", default=0,
            type=int, metavar="N");
    parser.add_argument("--input_mesh", default="model.obj", type=str);
    parser.add_argument("--output_mesh", default="model_out.obj", type=str);
    args = parser.parse_args();
    return args;


def main():
    args = parse_args();
    mesh = pymesh.load_mesh(args.input_mesh);
    mesh2 = trimesh.load(args.input_mesh)
    points, sdf = sample_sdf_near_surface(mesh2, number_of_points=4000, min_size=0.04)
    grid = pymesh.VoxelGrid(args.cell_size, mesh.dim);
    grid.insert_mesh(mesh);
    grid.create_grid();
    grid.dilate(args.dilate);
    grid.erode(args.erode);
    out_mesh = grid.mesh;
    pymesh.save_mesh(args.output_mesh, out_mesh);

if __name__ == "__main__":
    main();