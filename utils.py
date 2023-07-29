import numpy as np
import math
import torch
import os


def chamfer_loss(points_src, points_tgt):
    ''' Computes minimal distances of each point in points_src to points_tgt.

    Args:
        points_src (numpy array): source points
        normals_src (numpy array): source normals
        points_tgt (numpy array): target points
        normals_tgt (numpy array): target normals
    '''
    dist_matrix = ((points_src.unsqueeze(2) - points_tgt.unsqueeze(1))**2).sum(-1)
    dist_complete = (dist_matrix.min(-1)[0]).mean(-1)
    dist_acc = (dist_matrix.min(-2)[0]).mean(-1)
    dist = ((dist_acc + dist_complete)/2).mean()*1e4
    return dist


def octree(point, x_boundary, y_boundary, z_boundary, size, octant_list=[], octant_info=[]):
    if point.shape[0] >= 50:
        x_index = point[:, 0] > x_boundary
        y_index = point[:, 1] > y_boundary
        z_index = point[:, 2] > z_boundary
        octant0 = point[torch.where(x_index & y_index & z_index)]
        octant1 = point[torch.where(x_index & ~y_index & z_index)]
        octant2 = point[torch.where(x_index & y_index & ~z_index)]
        octant3 = point[torch.where(x_index & ~y_index & ~z_index)]
        octant4 = point[torch.where(~x_index & ~y_index & z_index)]
        octant5 = point[torch.where(~x_index & y_index & ~z_index)]
        octant6 = point[torch.where(~x_index & ~y_index & ~z_index)]
        octant7 = point[torch.where(~x_index & y_index & z_index)]

        # octant0 = point[np.where((point[:,0] > x_boundary-size/4) & (point[:,1] > y_boundary-size/4) & (point[:,2] > z_boundary-size/4))]
        # octant1 = point[np.where((point[:,0] > x_boundary-size/4) & (point[:,1] < y_boundary+size/4) & (point[:,2] > z_boundary-size/4))]
        # octant2 = point[np.where((point[:,0] > x_boundary-size/4) & (point[:,1] > y_boundary-size/4) & (point[:,2] < z_boundary+size/4))]
        # octant3 = point[np.where((point[:,0] > x_boundary-size/4) & (point[:,1] < y_boundary+size/4) & (point[:,2] < z_boundary+size/4))]
        # octant4 = point[np.where((point[:,0] < x_boundary+size/4) & (point[:,1] < y_boundary+size/4) & (point[:,2] > z_boundary-size/4))]
        # octant5 = point[np.where((point[:,0] < x_boundary+size/4) & (point[:,1] > y_boundary-size/4) & (point[:,2] < z_boundary+size/4))]
        # octant6 = point[np.where((point[:,0] < x_boundary+size/4) & (point[:,1] < y_boundary+size/4) & (point[:,2] < z_boundary+size/4))]
        # octant7 = point[np.where((point[:,0] < x_boundary+size/4) & (point[:,1] > y_boundary-size/4) & (point[:,2] > z_boundary-size/4))]

        percent = 1 / 4
        octree(octant0, x_boundary + size * percent, y_boundary + size * percent, z_boundary + size * percent, size / 2,
               octant_list, octant_info)
        octree(octant1, x_boundary + size * percent, y_boundary - size * percent, z_boundary + size * percent, size / 2,
               octant_list, octant_info)
        octree(octant2, x_boundary + size * percent, y_boundary + size * percent, z_boundary - size * percent, size / 2,
               octant_list, octant_info)
        octree(octant3, x_boundary + size * percent, y_boundary - size * percent, z_boundary - size * percent, size / 2,
               octant_list, octant_info)
        octree(octant4, x_boundary - size * percent, y_boundary - size * percent, z_boundary + size * percent, size / 2,
               octant_list, octant_info)
        octree(octant5, x_boundary - size * percent, y_boundary + size * percent, z_boundary - size * percent, size / 2,
               octant_list, octant_info)
        octree(octant6, x_boundary - size * percent, y_boundary - size * percent, z_boundary - size * percent, size / 2,
               octant_list, octant_info)
        octree(octant7, x_boundary - size * percent, y_boundary + size * percent, z_boundary + size * percent, size / 2,
               octant_list, octant_info)

    elif 50 > point.shape[0] > 0:
        octant_list.append(point)
        octant_info.append(np.array([x_boundary, y_boundary, z_boundary, size]))

    return octant_list, octant_info


def get_extended_point(point, octant_info, extend_size=1):
    point_list = []
    for octant in octant_info:
        center = octant[0:3]
        size = octant[3] / 2 * extend_size
        left_bottom = center - size
        right_up = center + size
        index = (point[:, 0] > left_bottom[0]) & (point[:, 1] > left_bottom[1])&(point[:, 2] > left_bottom[2]) &\
                (right_up[0] >= point[:, 0]) & (right_up[1] >= point[:, 1]) & (right_up[2] >= point[:, 2])
        index = torch.where(index)
        chosen_point = point[index]
        point_list.append(chosen_point)
    return point_list


def get_extended_point_with_label(point, octant_info, extend_size=1):
    point_list = []
    label_list = []
    for octant in octant_info:
        center = octant[0:3]
        size = octant[3] / 2 * extend_size
        left_bottom = center - size
        right_up = center + size
        index = (point[:, 0] > left_bottom[0]) & (point[:, 1] > left_bottom[1])&(point[:, 2] > left_bottom[2]) &\
                (right_up[0] >= point[:, 0]) & (right_up[1] >= point[:, 1]) & (right_up[2] >= point[:, 2])
        index = torch.where(index)
        temp = point[index]
        chosen_point = temp[:, 0:3]
        chosen_label = temp[:, 3]
        point_list.append(chosen_point)
        label_list.append(chosen_label)
    return point_list, label_list


def rotation(theta, axis=1):
    matrix = []
    matrix.append(np.array([math.cos(theta), 0, math.sin(theta)]))
    matrix.append(np.array([0, 1, 0]))
    matrix.append(np.array([math.sin(-theta), 0, math.cos(theta)]))
    return torch.from_numpy(np.array(matrix).T).float()


def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids


def square_distance(src, dst):
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


def knn_point(nsample, xyz, new_xyz, dis=False):
    """
    Input:
        nsample: max sample number in local region
        xyz: all points, [B, N, C]
        new_xyz: query points, [B, S, C]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    sqrdists = square_distance(new_xyz, xyz)
    g_dis, group_idx = torch.topk(sqrdists, nsample, dim=-1, largest=False, sorted=True)
    if dis:
        return  group_idx, g_dis
    return group_idx


def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


# .ply format  --  X,Y,Z, normalX,normalY,normalZ
def parse_ply_planes(shape_name, num_of_points=2048):
    file = open(shape_name, 'r')
    lines = file.readlines()
    vertices = np.zeros([num_of_points, 7], np.float32)
    assert lines[9].strip() == "end_header"
    for i in range(num_of_points):
        line = lines[i + 10].split()
        vertices[i, 0] = float(line[0])  # X
        vertices[i, 1] = float(line[1])  # Y
        vertices[i, 2] = float(line[2])  # Z
        vertices[i, 3] = float(line[3])  # normalX
        vertices[i, 4] = float(line[4])  # normalY
        vertices[i, 5] = float(line[5])  # normalZ
        tmp = vertices[i, 0] * vertices[i, 3] + vertices[i, 1] * vertices[i, 4] + vertices[i, 2] * vertices[i, 5]
        vertices[i, 6] = -tmp  # d for plane ax+by+cz+d = 0
    return vertices


def parse_ply_list_to_planes(ref_txt_name, data_dir, data_txt_name):
    # open file & read points
    ref_file = open(ref_txt_name, 'r')
    ref_names = [line.strip() for line in ref_file]
    ref_file.close()
    data_file = open(data_txt_name, 'r')
    data_names = [line.strip() for line in data_file]
    data_file.close()

    num_shapes = len(ref_names)
    ref_points = np.zeros([num_shapes, 2048, 7], np.float32)
    idx = np.zeros([num_shapes], np.int32)

    for i in range(num_shapes):
        shape_name = data_dir + "/" + ref_names[i] + ".ply"
        shape_idx = data_names.index(ref_names[i])
        shape_planes = parse_ply_planes(shape_name)
        ref_points[i, :, :] = shape_planes
        idx[i] = shape_idx

    return ref_points, idx, ref_names


def write_ply_point(name, vertices):
    fout = open(name, 'w')
    fout.write("ply\n")
    fout.write("format ascii 1.0\n")
    fout.write("element vertex " + str(len(vertices)) + "\n")
    fout.write("property float x\n")
    fout.write("property float y\n")
    fout.write("property float z\n")
    fout.write("end_header\n")
    for ii in range(len(vertices)):
        fout.write(str(vertices[ii, 0]) + " " + str(vertices[ii, 1]) + " " + str(vertices[ii, 2]) + "\n")
    fout.close()


def write_ply_point_normal(name, vertices, normals=None):
    fout = open(name, 'w')
    fout.write("ply\n")
    fout.write("format ascii 1.0\n")
    fout.write("element vertex " + str(len(vertices)) + "\n")
    fout.write("property float x\n")
    fout.write("property float y\n")
    fout.write("property float z\n")
    fout.write("property float nx\n")
    fout.write("property float ny\n")
    fout.write("property float nz\n")
    fout.write("end_header\n")
    if normals is None:
        for ii in range(len(vertices)):
            fout.write(str(vertices[ii, 0]) + " " + str(vertices[ii, 1]) + " " + str(vertices[ii, 2]) + " " + str(
                vertices[ii, 3]) + " " + str(vertices[ii, 4]) + " " + str(vertices[ii, 5]) + "\n")
    else:
        for ii in range(len(vertices)):
            fout.write(str(vertices[ii, 0]) + " " + str(vertices[ii, 1]) + " " + str(vertices[ii, 2]) + " " + str(
                normals[ii, 0]) + " " + str(normals[ii, 1]) + " " + str(normals[ii, 2]) + "\n")
    fout.close()


def write_ply_triangle(name, vertices, triangles):
    fout = open(name, 'w')
    fout.write("ply\n")
    fout.write("format ascii 1.0\n")
    fout.write("element vertex " + str(len(vertices)) + "\n")
    fout.write("property float x\n")
    fout.write("property float y\n")
    fout.write("property float z\n")
    fout.write("element face " + str(len(triangles)) + "\n")
    fout.write("property list uchar int vertex_index\n")
    fout.write("end_header\n")
    for ii in range(len(vertices)):
        fout.write(str(vertices[ii, 0]) + " " + str(vertices[ii, 1]) + " " + str(vertices[ii, 2]) + "\n")
    for ii in range(len(triangles)):
        fout.write("3 " + str(triangles[ii, 0]) + " " + str(triangles[ii, 1]) + " " + str(triangles[ii, 2]) + "\n")
    fout.close()


def write_ply_polygon(name, vertices, polygons):
    fout = open(name, 'w')
    fout.write("ply\n")
    fout.write("format ascii 1.0\n")
    fout.write("element vertex " + str(len(vertices)) + "\n")
    fout.write("property float x\n")
    fout.write("property float y\n")
    fout.write("property float z\n")
    fout.write("element face " + str(len(polygons)) + "\n")
    fout.write("property list uchar int vertex_index\n")
    fout.write("end_header\n")
    for ii in range(len(vertices)):
        fout.write(str(vertices[ii][0]) + " " + str(vertices[ii][1]) + " " + str(vertices[ii][2]) + "\n")
    for ii in range(len(polygons)):
        fout.write(str(len(polygons[ii])))
        for jj in range(len(polygons[ii])):
            fout.write(" " + str(polygons[ii][jj]))
        fout.write("\n")
    fout.close()


def read_ply_polygon(name):
    fin = open(name, 'r')
    content = fin.readlines()
    vertices_len = int(content[2][15:-1])
    polygons_len = int(content[6][13:-1])
    vertices = content[9:9 + vertices_len]
    polygons = content[9 + vertices_len: 9 + vertices_len + polygons_len]
    fin.close()
    temp_file = "current.txt"
    fout = open(temp_file, 'w')
    fout.writelines(vertices)
    fout.flush()
    vertices = np.loadtxt(temp_file)
    fout = open(temp_file, 'w')
    fout.writelines(polygons)
    fout.flush()
    polygons = np.loadtxt(temp_file).astype(np.int)
    fout.close()
    return vertices, polygons[:, 1:4]


def write_obj_triangle(name, vertices, triangles):
    fout = open(name, 'w')
    for ii in range(len(vertices)):
        fout.write("v " + str(vertices[ii, 0]) + " " + str(vertices[ii, 1]) + " " + str(vertices[ii, 2]) + "\n")
    for ii in range(len(triangles)):
        fout.write(
            "f " + str(triangles[ii, 0] + 1) + " " + str(triangles[ii, 1] + 1) + " " + str(triangles[ii, 2] + 1) + "\n")
    fout.close()


def write_obj_polygon(name, vertices, polygons):
    fout = open(name, 'w')
    for ii in range(len(vertices)):
        fout.write("v " + str(vertices[ii][0]) + " " + str(vertices[ii][1]) + " " + str(vertices[ii][2]) + "\n")
    for ii in range(len(polygons)):
        fout.write("f")
        for jj in range(len(polygons[ii])):
            fout.write(" " + str(polygons[ii][jj] + 1))
        fout.write("\n")
    fout.close()


# designed to take 64^3 voxels!
def sample_points_polygon_vox64(vertices, polygons, voxel_model_64, num_of_points):
    # convert polygons to triangles
    triangles = []
    for ii in range(len(polygons)):
        for jj in range(len(polygons[ii]) - 2):
            triangles.append([polygons[ii][0], polygons[ii][jj + 1], polygons[ii][jj + 2]])
    triangles = np.array(triangles, np.int32)
    vertices = np.array(vertices, np.float32)

    small_step = 1.0 / 64
    epsilon = 1e-6
    triangle_area_list = np.zeros([len(triangles)], np.float32)
    triangle_normal_list = np.zeros([len(triangles), 3], np.float32)
    for i in range(len(triangles)):
        # area = |u x v|/2 = |u||v|sin(uv)/2
        a, b, c = vertices[triangles[i, 1]] - vertices[triangles[i, 0]]
        x, y, z = vertices[triangles[i, 2]] - vertices[triangles[i, 0]]
        ti = b * z - c * y
        tj = c * x - a * z
        tk = a * y - b * x
        area2 = math.sqrt(ti * ti + tj * tj + tk * tk)
        if area2 < epsilon:
            triangle_area_list[i] = 0
            triangle_normal_list[i, 0] = 0
            triangle_normal_list[i, 1] = 0
            triangle_normal_list[i, 2] = 0
        else:
            triangle_area_list[i] = area2
            triangle_normal_list[i, 0] = ti / area2
            triangle_normal_list[i, 1] = tj / area2
            triangle_normal_list[i, 2] = tk / area2

    triangle_area_sum = np.sum(triangle_area_list)
    sample_prob_list = (num_of_points / triangle_area_sum) * triangle_area_list

    triangle_index_list = np.arange(len(triangles))

    point_normal_list = np.zeros([num_of_points, 6], np.float32)
    count = 0
    watchdog = 0

    while (count < num_of_points):
        np.random.shuffle(triangle_index_list)
        watchdog += 1
        if watchdog > 100:
            print("infinite loop here!")
            return point_normal_list
        for i in range(len(triangle_index_list)):
            if count >= num_of_points: break
            dxb = triangle_index_list[i]
            prob = sample_prob_list[dxb]
            prob_i = int(prob)
            prob_f = prob - prob_i
            if np.random.random() < prob_f:
                prob_i += 1
            normal_direction = triangle_normal_list[dxb]
            u = vertices[triangles[dxb, 1]] - vertices[triangles[dxb, 0]]
            v = vertices[triangles[dxb, 2]] - vertices[triangles[dxb, 0]]
            base = vertices[triangles[dxb, 0]]
            for j in range(prob_i):
                # sample a point here:
                u_x = np.random.random()
                v_y = np.random.random()
                if u_x + v_y >= 1:
                    u_x = 1 - u_x
                    v_y = 1 - v_y
                ppp = u * u_x + v * v_y + base

                # verify normal
                pppn1 = (ppp + normal_direction * small_step + 0.5) * 64
                px1 = int(pppn1[0])
                py1 = int(pppn1[1])
                pz1 = int(pppn1[2])

                ppx = int((ppp[0] + 0.5) * 64)
                ppy = int((ppp[1] + 0.5) * 64)
                ppz = int((ppp[2] + 0.5) * 64)

                if ppx < 0 or ppx >= 64 or ppy < 0 or ppy >= 64 or ppz < 0 or ppz >= 64:
                    continue
                if voxel_model_64[
                    ppx, ppy, ppz] > 1e-3 or px1 < 0 or px1 >= 64 or py1 < 0 or py1 >= 64 or pz1 < 0 or pz1 >= 64 or \
                        voxel_model_64[px1, py1, pz1] > 1e-3:
                    # valid
                    point_normal_list[count, :3] = ppp
                    point_normal_list[count, 3:] = normal_direction
                    count += 1
                    if count >= num_of_points: break

    return point_normal_list


def sample_points_polygon(vertices, polygons, num_of_points):
    # convert polygons to triangles
    triangles = []
    for ii in range(len(polygons)):
        for jj in range(len(polygons[ii]) - 2):
            triangles.append([polygons[ii][0], polygons[ii][jj + 1], polygons[ii][jj + 2]])
    triangles = np.array(triangles, np.int32)
    vertices = np.array(vertices, np.float32)

    small_step = 1.0 / 64
    epsilon = 1e-6
    triangle_area_list = np.zeros([len(triangles)], np.float32)
    triangle_normal_list = np.zeros([len(triangles), 3], np.float32)
    for i in range(len(triangles)):
        # area = |u x v|/2 = |u||v|sin(uv)/2
        a, b, c = vertices[triangles[i, 1]] - vertices[triangles[i, 0]]
        x, y, z = vertices[triangles[i, 2]] - vertices[triangles[i, 0]]
        ti = b * z - c * y
        tj = c * x - a * z
        tk = a * y - b * x
        area2 = math.sqrt(ti * ti + tj * tj + tk * tk)
        if area2 < epsilon:
            triangle_area_list[i] = 0
            triangle_normal_list[i, 0] = 0
            triangle_normal_list[i, 1] = 0
            triangle_normal_list[i, 2] = 0
        else:
            triangle_area_list[i] = area2
            triangle_normal_list[i, 0] = ti / area2
            triangle_normal_list[i, 1] = tj / area2
            triangle_normal_list[i, 2] = tk / area2

    triangle_area_sum = np.sum(triangle_area_list)
    sample_prob_list = (num_of_points / triangle_area_sum) * triangle_area_list

    triangle_index_list = np.arange(len(triangles))

    point_normal_list = np.zeros([num_of_points, 6], np.float32)
    count = 0
    watchdog = 0

    while (count < num_of_points):
        np.random.shuffle(triangle_index_list)
        watchdog += 1
        if watchdog > 100:
            print("infinite loop here!")
            return point_normal_list
        for i in range(len(triangle_index_list)):
            if count >= num_of_points: break
            dxb = triangle_index_list[i]
            prob = sample_prob_list[dxb]
            prob_i = int(prob)
            prob_f = prob - prob_i
            if np.random.random() < prob_f:
                prob_i += 1
            normal_direction = triangle_normal_list[dxb]
            u = vertices[triangles[dxb, 1]] - vertices[triangles[dxb, 0]]
            v = vertices[triangles[dxb, 2]] - vertices[triangles[dxb, 0]]
            base = vertices[triangles[dxb, 0]]
            for j in range(prob_i):
                # sample a point here:
                u_x = np.random.random()
                v_y = np.random.random()
                if u_x + v_y >= 1:
                    u_x = 1 - u_x
                    v_y = 1 - v_y
                point_normal_list[count, :3] = u * u_x + v * v_y + base
                point_normal_list[count, 3:] = normal_direction
                count += 1
                if count >= num_of_points: break

    return point_normal_list


def sample_points(vertices, triangles, num_of_points):
    epsilon = 1e-6
    triangle_area_list = np.zeros([len(triangles)], np.float32)
    triangle_normal_list = np.zeros([len(triangles), 3], np.float32)
    for i in range(len(triangles)):
        # area = |u x v|/2 = |u||v|sin(uv)/2
        a, b, c = vertices[triangles[i, 1]] - vertices[triangles[i, 0]]
        x, y, z = vertices[triangles[i, 2]] - vertices[triangles[i, 0]]
        ti = b * z - c * y
        tj = c * x - a * z
        tk = a * y - b * x
        area2 = math.sqrt(ti * ti + tj * tj + tk * tk)
        if area2 < epsilon:
            triangle_area_list[i] = 0
            triangle_normal_list[i, 0] = 0
            triangle_normal_list[i, 1] = 0
            triangle_normal_list[i, 2] = 0
        else:
            triangle_area_list[i] = area2
            triangle_normal_list[i, 0] = ti / area2
            triangle_normal_list[i, 1] = tj / area2
            triangle_normal_list[i, 2] = tk / area2

    triangle_area_sum = np.sum(triangle_area_list)
    sample_prob_list = (num_of_points / triangle_area_sum) * triangle_area_list

    triangle_index_list = np.arange(len(triangles))

    point_normal_list = np.zeros([num_of_points, 6], np.float32)
    count = 0
    watchdog = 0

    while (count < num_of_points):
        np.random.shuffle(triangle_index_list)
        watchdog += 1
        if watchdog > 100:
            print("infinite loop here!")
            exit(0)
        for i in range(len(triangle_index_list)):
            if count >= num_of_points: break
            dxb = triangle_index_list[i]
            prob = sample_prob_list[dxb]
            prob_i = int(prob)
            prob_f = prob - prob_i
            if np.random.random() < prob_f:
                prob_i += 1
            normal_direction = triangle_normal_list[dxb]
            u = vertices[triangles[dxb, 1]] - vertices[triangles[dxb, 0]]
            v = vertices[triangles[dxb, 2]] - vertices[triangles[dxb, 0]]
            base = vertices[triangles[dxb, 0]]
            for j in range(prob_i):
                # sample a point here:
                u_x = np.random.random()
                v_y = np.random.random()
                if u_x + v_y >= 1:
                    u_x = 1 - u_x
                    v_y = 1 - v_y
                point_normal_list[count, :3] = (u * u_x + v * v_y + base)
                point_normal_list[count, 3:] = normal_direction
                count += 1
                if count >= num_of_points: break

    return point_normal_list


def load_data(txt_list, dxb, dataset_path=""):
    point_cloud = []
    points_chamfer = []
    query_p = []
    query_l = []
    clean_point = []
    chosen_num = 10000
    for idx in dxb:
        pre_path = os.path.join(dataset_path, txt_list[idx])
        pc = np.load(os.path.join(pre_path, "pointcloud.npz"))
        points = pc["points"]
        chamfer_choice = torch.randperm(points.shape[0])[0:20000].numpy()
        recon_choice = torch.randperm(points.shape[0])[0:3000].numpy()
        points_large = points[chamfer_choice]
        points = points[recon_choice]

        noise = 0.005 * np.random.randn(*points.shape)
        noise = noise.astype(np.float32)
        points_noise = points + noise

        point_cloud.append(points_noise)
        points_chamfer.append(points_large)
        train_points = np.load(os.path.join(pre_path, "points_train.npz"))
        clean_pc, points_surface, points_uniform, occupancies_surface, occupancies_uniform = \
            train_points["point_cloud"], train_points["points_surface"], train_points["points_uniform"], train_points["occupancies_surface"], \
            train_points["occupancies_uniform"]
        clean_pc -= 0.5

        # index = np.where(occupancies_surface < 0)
        # ddd = points_surface[index]
        # np.savetxt("label.txt", ddd, delimiter=";")
        # np.savetxt("input.txt", clean_pc, delimiter=";")
        # exit(0)
        # np.savetxt("sss.txt", clean_pc-0.5, delimiter=";")
        # np.savetxt("ccc.txt", points, delimiter=";")
        # np.savetxt("aaa.txt", points_surface, delimiter=";")
        # exit(0)
        surface_choice = torch.randperm(points_surface.shape[0])[0:6000].numpy()
        uniform_choice = torch.randperm(points_uniform.shape[0])[0:2000].numpy()
        points_surface = points_surface[surface_choice, :]
        points_uniform = points_uniform[uniform_choice, :]
        occupancies_surface = occupancies_surface[surface_choice]
        occupancies_uniform = occupancies_uniform[uniform_choice]
        query_p.append(np.concatenate([points_uniform, points_surface], 0))
        query_l.append(np.concatenate([occupancies_uniform, occupancies_surface], 0))
        clean_point.append(points)
    point_cloud = np.stack(point_cloud).astype(np.float32)
    clean_point = np.stack(clean_point).astype(np.float32)
    points_chamfer = np.stack(points_chamfer)
    query_p = np.stack(query_p).astype(np.float32)
    query_l = np.stack(query_l).astype(np.float32)

    # np.savetxt("gt.txt", query_p[0][np.where(query_l[0]>0.5)], delimiter=";")
    # np.savetxt("pc.txt", point_cloud[0], delimiter=";")
    # print(txt_list[dxb[0]])
    # exit(0)
    query_p = torch.from_numpy(query_p)
    query_l = torch.from_numpy(query_l)
    point_cloud = torch.from_numpy(point_cloud)[:, :, 0:3]
    clean_point = torch.from_numpy(clean_point)[:, :, 0:3]
    points_chamfer = torch.from_numpy(points_chamfer)[:, :, 0:3]
    # l = chamfer_loss(point_cloud, points_chamfer)
    return point_cloud, query_p, query_l, clean_point, points_chamfer


def load_data_test_iou(txt_list, dxb, dataset_path=""):
    point_cloud = []
    query_p = []
    query_l = []
    clean_point = []
    chosen_num = 10000
    for idx in dxb:
        pre_path = os.path.join(dataset_path, txt_list[idx])
        train_points = np.load(os.path.join(pre_path, "points_train.npz"))
        clean_pc, points_surface, points_uniform, occupancies_surface, occupancies_uniform = \
            train_points["point_cloud"], train_points["points_surface"], train_points["points_uniform"], train_points[
                "occupancies_surface"], \
            train_points["occupancies_uniform"]
        clean_pc -= 0.5

        pc = np.load(os.path.join(pre_path, "pointcloud.npz"))
        points = pc["points"]
        chamfer_choice = torch.randperm(points.shape[0])[0:3000].numpy()
        recon_choice = torch.randperm(points.shape[0])[0:3000].numpy()
        points_large = points[chamfer_choice]
        points = points[recon_choice]
        # indices = np.random.randint(points.shape[0], size=3000)
        # points = points[indices, :]
        noise = 0.005 * np.random.randn(*points.shape)
        noise = noise.astype(np.float32)
        points_noise = points + noise
        clean_point.append(points)
        point_cloud.append(points_noise)
        train_points = np.load(os.path.join(pre_path, "points.npz"))
        points, occupancies = train_points["points"], np.unpackbits(train_points["occupancies"])
        query_p.append(points)
        query_l.append(occupancies)
    point_cloud = np.stack(point_cloud).astype(np.float32)
    clean_point = np.stack(clean_point).astype(np.float32)
    query_p = np.stack(query_p).astype(np.float32)
    query_l = np.stack(query_l).astype(np.float32)
    # np.savetxt("gt.txt", query_p[0][np.where(query_l[0]>0.5)], delimiter=";")
    # np.savetxt("pc.txt", point_cloud[0], delimiter=";")
    # print(txt_list[dxb[0]])
    # exit(0)
    query_p = torch.from_numpy(query_p)
    query_l = torch.from_numpy(query_l)
    point_cloud = torch.from_numpy(point_cloud)[:, :, 0:3]
    clean_point = torch.from_numpy(clean_point)[:, :, 0:3]
    return point_cloud, query_p, query_l, clean_point


def load_data_test(txt_list, dxb, dataset_path=""):
    point_cloud = []
    loc_l = []
    scale_l = []
    clean_point = []
    chosen_num = 10000
    for idx in dxb:
        pre_path = os.path.join(dataset_path, txt_list[idx])
        pc = np.load(os.path.join(pre_path, "pointcloud.npz"))
        train_points = np.load(os.path.join(pre_path, "points_train.npz"))

        clean_pc, points_surface, points_uniform, occupancies_surface, occupancies_uniform = \
            train_points["point_cloud"], train_points["points_surface"], train_points["points_uniform"], train_points[
                "occupancies_surface"], \
            train_points["occupancies_uniform"]
        clean_pc -= 0.5

        loc_l.append(pc["loc"])
        points = pc["points"]
        point_choice = torch.randperm(points.shape[0])[0:3000].numpy()
        points = points[point_choice, :]
        noise = 0.002 * np.random.randn(*points.shape)
        noise = noise.astype(np.float32)
        points_noise = points + noise
        clean_point.append(points.astype(np.float32))
        point_cloud.append(points_noise)
    point_cloud = np.stack(point_cloud)
    clean_point = np.stack(clean_point)
    loc_l = np.stack(loc_l)
    point_cloud = torch.from_numpy(point_cloud)
    clean_point = torch.from_numpy(clean_point)
    loc_l = torch.from_numpy(loc_l)
    return point_cloud


def ABC_load_util(txt_list, dxb, dataset_path=""):
    point_cloud = []
    query_p = []
    query_l = []
    query_s = []
    for idx in dxb:
        pre_path = os.path.join(dataset_path, txt_list[idx])
        pc = np.load(os.path.join(pre_path, "pointcloud.npz"))
        points = pc["points"]
        choice = torch.randperm(points.shape[0])[0:3000].numpy()
        points = points[choice]
        point_cloud.append(points)
        train_points = np.load(os.path.join(pre_path, "points.npz"))
        query_point, query_label, query_sdf = train_points["points"], train_points["occupancies"], train_points["sdf"]

        query_label = np.unpackbits(query_label)
        query_p.append(query_point)
        query_l.append(query_label)
        query_s.append(query_sdf)
    point_cloud = np.stack(point_cloud).astype(np.float32)
    query_p = np.stack(query_p).astype(np.float32)
    query_l = np.stack(query_l).astype(np.float32)
    query_s = np.stack(query_s).astype(np.float32)

    query_p = torch.from_numpy(query_p)
    query_l = torch.from_numpy(query_l)
    query_s = torch.from_numpy(query_s)
    point_cloud = torch.from_numpy(point_cloud)[:, :, 0:3]
    # l = chamfer_loss(point_cloud, points_chamfer)
    return point_cloud, query_p, query_l, query_s, point_cloud


def ABC_load_util_test(txt_list, dxb, dataset_path=""):
    point_cloud = []
    query_p = []
    query_l = []
    query_s = []
    for idx in dxb:
        pre_path = os.path.join(dataset_path, txt_list[idx])
        pc = np.load(os.path.join(pre_path, "pointcloud.npz"))
        points = pc["points"]
        center = (points.max(0) + points.min(0))/2
        scale = (points.max(0) - points.min(0)).max()
        choice = torch.randperm(points.shape[0])[0:3000].numpy()
        points = points[choice]
        poistn = (points - center)/scale
        point_cloud.append(points)

    point_cloud = np.stack(point_cloud).astype(np.float32)
    point_cloud = torch.from_numpy(point_cloud)[:, :, 0:3]
    # l = chamfer_loss(point_cloud, points_chamfer)
    return point_cloud